from __future__ import annotations

import re
from typing import Sequence

from pydantic import BaseModel, Field, field_validator, model_validator

from chatlib.chatbot import Dialogue, DialogueTurn
from chatlib.chatbot import ChatCompletionParams
from chatlib.chatbot.generators import StateBasedResponseGenerator
from chatlib.llm.integration.gemini_api import GeminiAPI
from chatlib.tool.converter import generate_pydantic_converter
from chatlib.tool.versatile_mapper import (
    ChatCompletionFewShotMapperParams,
    DialogueSummarizer,
    MapperInputOutputPair,
)
from chatlib.utils.jinja_utils import convert_to_jinja_template

from app.common import PromptFactory
from app.gemini_generator import GeminiGenerator


class TutorGenerator(GeminiGenerator):
    def __init__(self):
        template = """
You are CHACHA, the same cheerful peer friend, now switching into a study buddy role.
Your job is to help the user make progress on a homework-style assignment while keeping them engaged and thinking deeply about math.

[Context]
- The user recently described feeling: {{ feeling_sentiment | default("unknown") }}.
- Their words about it: {{ feeling_statement | default("not provided") }}.
- Initial assignment prompt: {{ assignment_initial_prompt | default("not provided") }}.
- Math question to tackle now: {{ assignment_request | default("awaiting details") }}.
- Tutor stage: {{ tutor_stage | default("collect_question") }}.

[Guiding principles]
- Stay friendly and encouraging while keeping the tone aligned with a curious classmate.
- Keep the conversation math-first: use numbers, quantities, and concrete reasoning whenever possible.
- Model active learning: ask what they understand, what they've tried, and what goal they are aiming for.
- Target higher levels of Bloom's taxonomy: analysis, evaluation, and creation. Prompt them to compare, justify, or extend ideas.
- Offer hints, scaffolding questions, and partial steps instead of full solutions. Never hand over the entire answer.
- Break problems into 3–5 bite-sized checkpoints. After each hint, ask the user to attempt the next step.
- Celebrate effort, acknowledge feelings, and tie support back to their earlier emotion when helpful.
- End turns by requesting the user to summarise their plan, explain reasoning, or try a next action.

{%- if tutor_stage == "collect_question" %}
[Current focus]
- They asked for help but have not pasted the exact math problem yet.
- Ask them politely to share the full question or example (include numbers, operations, what needs solving).
- Remind them you will break it into 3–5 friendly steps once you see the full math question.
- Do **not** start solving yet; stay encouraging and inquisitive.
{%- elif tutor_stage == "quest_ready" %}
[Current focus]
- They shared this problem: {{ assignment_request | default("unknown math question") }}.
- Walk through the thinking in 3–5 clear math steps before inviting them to play the quest.
- Highlight what each checkpoint covers and why it helps solve the problem.
- Encourage them to tap Play to practice; mention the quest gives instant feedback, hints, and helpful bridge challenges.
{%- elif tutor_stage == "quest_summary" %}
[Quest reflection]
- The quest is complete. Use the summary data to celebrate effort and highlight learning.
{%- if tutor_summary %}
  • Score: {{ tutor_summary.score }}.
  • Badges earned: {{ tutor_summary.badges | join(", ") if tutor_summary.badges else "none" }}.
  • Levels cleared: {{ tutor_summary.completed_levels }}/{{ tutor_summary.total_levels }}.
{%- endif %}
- Recap the main 3–5 math moves that solve the original question.
- Note one next action (practice variation, check work, ask a follow-up) and invite them to share how they feel about it now.
{%- endif %}

{%- if tutor_plan %}
[Study quest integration]
- The system already generated a quest with {{ tutor_plan.steps|length }} levels:
{%- for step in tutor_plan.steps %}
  • Level {{ loop.index }} – {{ step.title }}: {{ step.objective }}
{%- endfor %}
- Mention how each level maps to the math steps you outlined and why the game will help them focus.
{%- endif %}

""" + PromptFactory.get_speaking_rules_block()
        super().__init__(base_instruction=template)

    async def _get_response_impl(self, dialogue: list[DialogueTurn], dry: bool = False):
        message, metadata = await super()._get_response_impl(dialogue, dry)
        enriched_metadata = dict(metadata or {})
        enriched_metadata.setdefault("tutor_mode", True)
        enriched_metadata.setdefault(
            "tutor_badge",
            {
                "title": "Study Buddy Mode",
                "subtitle": "Let’s tackle your assignment step by step!",
            },
        )
        tutor_plan = self._params.get("tutor_plan")
        if tutor_plan:
            enriched_metadata.setdefault("tutor_game_ready", True)
            enriched_metadata.setdefault("tutor_game_plan", tutor_plan)
        tutor_stage = self._params.get("tutor_stage")
        if tutor_stage is None:
            tutor_stage = "quest_ready" if tutor_plan else "collect_question"
        enriched_metadata.setdefault("tutor_stage", tutor_stage)
        assignment_prompt = self._params.get("assignment_request") or self._params.get("assignment_initial_prompt")
        if assignment_prompt:
            enriched_metadata.setdefault("tutor_assignment_prompt", assignment_prompt)
        return message, enriched_metadata


def create_generator():
    return TutorGenerator()


class TutorBridgeStep(BaseModel):
    prompt: str
    hint: str
    success_keywords: list[str] = Field(default_factory=list)

    @field_validator("prompt", "hint", mode="before")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        return value.strip() if isinstance(value, str) else value

    @field_validator("success_keywords", mode="before")
    @classmethod
    def _normalize_keywords(cls, value: Sequence[str] | str | None) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            candidates = re.split(r"[,\n]", value)
        else:
            candidates = value
        keywords: list[str] = []
        for candidate in candidates:
            token = (candidate or "").strip().lower()
            if token and token not in keywords:
                keywords.append(token)
        return keywords[:4]


class TutorGameStep(BaseModel):
    id: str
    title: str
    objective: str
    check_prompt: str
    success_keywords: list[str] = Field(default_factory=list)
    hint: str
    bridge: TutorBridgeStep

    @field_validator("id", "title", "objective", "check_prompt", "hint", mode="before")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        return value.strip() if isinstance(value, str) else value

    @field_validator("success_keywords", mode="before")
    @classmethod
    def _normalize_keywords(cls, value: Sequence[str] | str | None) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            candidates = re.split(r"[,\n]", value)
        else:
            candidates = value
        keywords: list[str] = []
        for candidate in candidates:
            token = (candidate or "").strip().lower()
            if token and token not in keywords:
                keywords.append(token)
        return keywords[:4]


class TutorGamePlan(BaseModel):
    assignment_overview: str
    steps: list[TutorGameStep]

    @model_validator(mode="after")
    def _validate_steps(self) -> "TutorGamePlan":
        if not 3 <= len(self.steps) <= 5:
            raise ValueError("steps must contain between 3 and 5 items")
        return self


class TutorPlanParams(ChatCompletionFewShotMapperParams):
    assignment_request: str
    feeling_statement: str | None = None
    feeling_sentiment: str | None = None
    user_name: str | None = None
    user_age: int | None = None
    locale: str | None = None
    model: str = "gemini-1.5-pro"
    api_params: ChatCompletionParams = ChatCompletionParams(temperature=0.2)


_PLAN_INSTRUCTION_TEMPLATE = convert_to_jinja_template(
    """
You are designing a playful "Study Quest" for a young student. Turn the assignment into a focused plan.

- Assignment: "{{ assignment_request }}"
{%- if feeling_statement %}
- The student recently said: "{{ feeling_statement }}"
{%- endif %}
- Mood cue: {{ feeling_sentiment or "unknown" }}.
- Age: {{ user_age or "unknown" }}. Locale: {{ locale or "en" }}. Keep tone upbeat and kid-friendly.
- Always frame the plan around working through a math example. If the request is vague, invent a simple, grade-appropriate math scenario that fits.

Return STRICT JSON (no markdown, no commentary) exactly matching:
{
  "assignment_overview": "<one upbeat sentence>",
  "steps": [
    {
      "id": "step-<number>",
      "title": "<2-4 word title>",
      "objective": "<=18 words, action oriented>",
      "check_prompt": "<question for the student to answer inside the game>",
      "success_keywords": ["<keyword1>", "<keyword2>", "..."],
      "hint": "<=15 words hint that nudges the right move>",
      "bridge": {
        "prompt": "<simpler question that breaks the task down>",
        "hint": "<=12 words of friendly encouragement>",
        "success_keywords": ["<keyword1>", "..."]
      }
    }
  ]
}

Rules:
- Create 3 to 5 total steps.
- Use lowercase keywords, 1-4 items per list, no duplicates.
- Each step must focus on one discrete micro-action.
- Bridge prompts must be simpler than the main prompt and guide the student back on track.
"""
)

_PLAN_EXAMPLES = [
    MapperInputOutputPair(
        input=[
            DialogueTurn(
                message="Can you help me write a persuasive paragraph about why our school should add more recycling bins?",
                is_user=True,
            )
        ],
        output={
            "assignment_overview": "We'll craft a punchy paragraph that convinces readers to add more recycling bins at school.",
            "steps": [
                {
                    "id": "step-1",
                    "title": "Choose Angle",
                    "objective": "Decide on the strongest reason to convince classmates about recycling bins.",
                    "check_prompt": "What main reason will you highlight in your paragraph?",
                    "success_keywords": ["reason", "main", "recycling"],
                    "hint": "Pick the reason that would persuade you most.",
                    "bridge": {
                        "prompt": "List one benefit students get from extra recycling bins.",
                        "hint": "Think about how it helps daily school life.",
                        "success_keywords": ["benefit", "bins", "students"],
                    },
                },
                {
                    "id": "step-2",
                    "title": "Support Proof",
                    "objective": "Gather one fact or example that backs up your main reason.",
                    "check_prompt": "Which fact or example will you use to prove your reason?",
                    "success_keywords": ["fact", "example", "proof"],
                    "hint": "Try a statistic, school observation, or trusted quote.",
                    "bridge": {
                        "prompt": "Name one real thing you noticed that shows recycling bins help.",
                        "hint": "Think about messes avoided or materials saved.",
                        "success_keywords": ["noticed", "bins", "help"],
                    },
                },
                {
                    "id": "step-3",
                    "title": "Call To Act",
                    "objective": "Write a closing sentence that urges readers to support more bins.",
                    "check_prompt": "How will you ask readers to take action or agree with you?",
                    "success_keywords": ["ask", "action", "support"],
                    "hint": "Use a friendly but clear action phrase.",
                    "bridge": {
                        "prompt": "Write one short phrase that tells classmates what to do next.",
                        "hint": "Use words like 'let’s' or 'we can'.",
                        "success_keywords": ["let", "we", "can"],
                    },
                },
            ],
        },
    )
]

_str_to_tutor_plan, _tutor_plan_to_str = generate_pydantic_converter(TutorGamePlan)


def _generate_instruction(dialogue: Dialogue, params: TutorPlanParams) -> str:
    return _PLAN_INSTRUCTION_TEMPLATE.render(
        assignment_request=params.assignment_request,
        feeling_statement=params.feeling_statement,
        feeling_sentiment=params.feeling_sentiment,
        user_age=params.user_age,
        locale=params.locale,
    )


plan_summarizer = DialogueSummarizer[TutorGamePlan, TutorPlanParams](
    api=GeminiAPI(),
    instruction_generator=_generate_instruction,
    output_str_converter=_tutor_plan_to_str,
    str_output_converter=_str_to_tutor_plan,
    dialogue_filter=lambda dialogue, params: StateBasedResponseGenerator.trim_dialogue_recent_n_states(
        dialogue, 1
    ),
)


def _extract_keywords(text: str, limit: int = 4) -> list[str]:
    tokens = re.findall(r"[a-zA-Z]{3,}", text.lower())
    seen: list[str] = []
    for token in tokens:
        if token not in seen:
            seen.append(token)
        if len(seen) >= limit:
            break
    return seen


def _ensure_keywords(existing: Sequence[str], source_text: str, fallback_token: str = "idea") -> list[str]:
    cleaned = [token.strip().lower() for token in existing if token.strip()]
    if cleaned:
        return list(dict.fromkeys(cleaned))[:4]
    extracted = _extract_keywords(source_text, limit=4)
    if extracted:
        return extracted[:4]
    return [fallback_token]


def _default_bridge_for(step: TutorGameStep) -> TutorBridgeStep:
    keywords_seed = f"{step.title} {step.objective}"
    return TutorBridgeStep(
        prompt=f"Name one tiny piece that helps with {step.title.lower()}.",
        hint="Zoom in on one detail you can handle right now.",
        success_keywords=_extract_keywords(keywords_seed, limit=2) or ["detail"],
    )


def normalize_tutor_plan(plan: TutorGamePlan, assignment_request: str) -> TutorGamePlan:
    normalized_steps: list[TutorGameStep] = []
    sliced_steps = plan.steps[:5]
    for index, step in enumerate(sliced_steps, start=1):
        keywords_source = f"{step.objective} {step.check_prompt}"
        bridge = step.bridge or _default_bridge_for(step)
        normalized_bridge = bridge.model_copy(
            update={
                "success_keywords": _ensure_keywords(
                    bridge.success_keywords, f"{step.title} {bridge.prompt}", fallback_token="clue"
                )
            }
        )
        normalized_step = step.model_copy(
            update={
                "id": step.id or f"step-{index}",
                "title": step.title or f"Checkpoint {index}",
                "success_keywords": _ensure_keywords(step.success_keywords, keywords_source),
                "bridge": normalized_bridge,
            }
        )
        normalized_steps.append(normalized_step)

    if len(normalized_steps) < 3:
        fallback = build_fallback_plan(assignment_request)
        for fallback_step in fallback.steps:
            if len(normalized_steps) >= 3:
                break
            normalized_steps.append(fallback_step)

    overview = (plan.assignment_overview or assignment_request).strip()
    return TutorGamePlan(assignment_overview=overview, steps=normalized_steps)


def build_fallback_plan(assignment_request: str) -> TutorGamePlan:
    assignment_focus = assignment_request.strip() or "this assignment"
    base_keywords = _extract_keywords(assignment_focus, limit=3) or ["plan", "draft", "review"]

    def _make_step(
        idx: int,
        title: str,
        objective: str,
        prompt: str,
        keywords: list[str],
        hint: str,
        bridge_prompt: str,
        bridge_hint: str,
    ) -> TutorGameStep:
        bridge = TutorBridgeStep(
            prompt=bridge_prompt,
            hint=bridge_hint,
            success_keywords=[keywords[0]] if keywords else ["idea"],
        )
        return TutorGameStep(
            id=f"step-{idx}",
            title=title,
            objective=objective,
            check_prompt=prompt,
            success_keywords=keywords,
            hint=hint,
            bridge=bridge,
        )

    steps = [
        _make_step(
            1,
            "Understand Problem",
            "Restate the question and list important numbers or units.",
            "What information do you know and what is the question asking you to find?",
            _ensure_keywords(base_keywords + ["numbers", "units"], assignment_focus),
            "Name the numbers and what they represent.",
            "Write the key numbers or quantities you see.",
            "List the numbers and labels in the problem.",
        ),
        _make_step(
            2,
            "Plan Strategy",
            "Decide which operation or equation will solve the problem.",
            "Which math steps or operation will you use to solve it?",
            _ensure_keywords(base_keywords + ["solve", "operation", "equation"], assignment_focus),
            "Pick the operation that connects the numbers.",
            "Write a mini equation that matches the story.",
            "Try writing the equation or operation that fits best.",
        ),
        _make_step(
            3,
            "Solve & Check",
            "Do the calculation and make sure the answer makes sense.",
            "What answer do you get and how can you check that it works?",
            _ensure_keywords(base_keywords + ["answer", "check"], assignment_focus),
            "Do the math, then check it another way.",
            "Try the calculation with easy numbers first.",
            "Use another quick check to see if the answer fits.",
        ),
    ]

    return TutorGamePlan(
        assignment_overview=f"Let's turn {assignment_focus} into a friendly math plan.",
        steps=steps,
    )


async def build_tutor_game_plan(dialogue: Dialogue, params: TutorPlanParams) -> TutorGamePlan:
    try:
        plan = await plan_summarizer.run(_PLAN_EXAMPLES, dialogue, params)
        return normalize_tutor_plan(plan, params.assignment_request)
    except Exception:
        return build_fallback_plan(params.assignment_request)
