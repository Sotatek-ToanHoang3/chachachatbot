from __future__ import annotations

import re
from typing import Any, Sequence

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
- Name how each checkpoint connects to a math operation or action (e.g. pick the operation, compute, check).
- Encourage them to tap Play to practice; mention the quest gives instant feedback, hints, and helpful bridge challenges.
{%- elif tutor_stage == "quest_summary" %}
[Quest reflection]
- The quest is complete. Use the summary data to celebrate effort and highlight learning.
{%- if tutor_summary %}
  • Score: {{ tutor_summary.score }}.
  • Badges earned: {{ tutor_summary.badges | join(", ") if tutor_summary.badges else "none" }}.
  • Levels cleared: {{ tutor_summary.completed_levels }}/{{ tutor_summary.total_levels }}.
{%- endif %}
- Clearly state the final numeric answer (with units) before anything else.
- Recap the main 3–5 math moves in order, calling out which operation each checkpoint used.
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


class TutorSubstep(BaseModel):
    label: str
    prompt: str
    memory_hook: str | None = None
    visual: str | None = None
    parallel_example: str | None = None

    @field_validator("label", "prompt", "memory_hook", "visual", "parallel_example", mode="before")
    @classmethod
    def _strip_text(cls, value: str | None) -> str | None:
        return value.strip() if isinstance(value, str) else value


class TutorGameStep(BaseModel):
    id: str
    title: str
    objective: str
    check_prompt: str
    success_keywords: list[str] = Field(default_factory=list)
    hint: str
    bridge: TutorBridgeStep
    substeps: list[TutorSubstep] = Field(default_factory=list)

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
        if not 2 <= len(self.steps) <= 5:
            raise ValueError("steps must contain between 2 and 5 items")
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
- Count the math operations in the assignment. When the story says something is shared equally and then some groups are removed (keywords: equally, sold, gave away, remaining), create exactly **two** checkpoints: (1) divide to find the per-group amount, (2) reuse it to compute what is left (subtract groups, multiply, state the final answer). Otherwise pick 3 checkpoints for single-step tasks, 4 for medium multi-step prompts, and 5 when several operations or sentences are involved.
- Order the checkpoints following the operations as they appear. If no operations are obvious, still create at least three steps by inventing a simple example and focus on strategy, computation, and checking.
- Each checkpoint must include three substeps: (1) a tiny action to keep momentum, (2) a memory hook that explicitly recalls the previous answer, (3) a parallel simpler example that includes at least one emoji or ASCII doodle.
- Make every substep prompt playful and visual with emojis or simple drawings so the learner can picture the move.

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
      "substeps": [
        {
          "label": "<emoji + 1-2 word label>",
          "prompt": "<=18 words, includes at least one emoji or ASCII sketch>",
          "memory_hook": "<explicit reminder of what earlier answer to reuse>",
          "visual": "<emoji or ASCII doodle of the idea>",
          "parallel_example": "<mini example with tiny numbers and emoji>"
        }
      ],
      "bridge": {
        "prompt": "<simpler question that breaks the task down>",
        "hint": "<=12 words of friendly encouragement>",
        "success_keywords": ["<keyword1>", "..."]
      }
    }
  ]
}

Rules:
- Create 2 to 5 total steps, following the difficulty guidance above.
- Use lowercase keywords, 1-4 items per list, no duplicates.
- Each step must focus on one discrete micro-action.
- Provide exactly three substeps per step following the structure above. The second substep must reference the previous answer or checkpoint by name.
- The final checkpoint must have the learner state the final answer (numbers plus units) before checking.
- Bridge prompts must be simpler than the main prompt and guide the student back on track.
- Every hint and bridge hint should begin with "Example:" or "Try:" and give a quick, concrete action or wording the learner can reuse.
"""
)

_PLAN_EXAMPLES = [
    MapperInputOutputPair(
        input=[
            DialogueTurn(
                message="Can you help me solve 48 ÷ 6 + 5? I mix up the steps.",
                is_user=True,
            )
        ],
        output={
            "assignment_overview": "We'll break 48 ÷ 6 + 5 into a playful quest with division then addition.",
            "steps": [
                {
                    "id": "step-1",
                    "title": "Scan The Task",
                    "objective": "Spot every number and math symbol before any calculation.",
                    "check_prompt": "Which numbers and symbols appear in 48 ÷ 6 + 5?",
                    "success_keywords": ["48", "6", "5", "symbols"],
                    "hint": "Example: list each number and symbol in order.",
                    "substeps": [
                        {
                            "label": "🔍 Spot",
                            "prompt": "🔍 Circle 48, 6, 5, plus the ÷ and + signs.",
                            "memory_hook": "Keep those exact numbers handy for Step 2.",
                            "visual": "48 ÷ 6 + 5",
                            "parallel_example": "Trace 20 ÷ 5 + 4 with tiny cubes 🧊🧊🧊🧊."
                        },
                        {
                            "label": "🧠 Recall",
                            "prompt": "🧠 Say aloud the question: what does 48 ÷ 6 + 5 equal?",
                            "memory_hook": "Repeat the full question so it sticks for the next step.",
                            "visual": "🧠➡️❓",
                            "parallel_example": "Whisper 'What is 12 ÷ 3 + 2?' 🤔"
                        },
                        {
                            "label": "🎨 Sketch",
                            "prompt": "🎨 Draw six boxes for the division and add five dots beside them.",
                            "memory_hook": "Glance at the sketch whenever you name the numbers later.",
                            "visual": "[ ][ ][ ][ ][ ][ ] + .....",
                            "parallel_example": "Sketch three boxes and add two dots 🎯"
                        }
                    ],
                    "bridge": {
                        "prompt": "Name the numbers and symbols in 20 ÷ 5 + 4.",
                        "hint": "Example: read the short equation slowly out loud.",
                        "success_keywords": ["20", "5", "4"],
                    },
                },
                {
                    "id": "step-2",
                    "title": "Divide First",
                    "objective": "Divide 48 by 6 to find the equal groups.",
                    "check_prompt": "What is 48 ÷ 6 and how do you know?",
                    "success_keywords": ["divide", "48", "6", "8"],
                    "hint": "Example: split 48 into six equal stacks.",
                    "substeps": [
                        {
                            "label": "🧩 Break",
                            "prompt": "🧩 Group 48 into six piles of 8 counters 🧊.",
                            "memory_hook": "Use the numbers you spotted during Step 1.",
                            "visual": "8+8+8+8+8+8",
                            "parallel_example": "Make 12 into three piles of 4 🧱🧱🧱."
                        },
                        {
                            "label": "🧠 Recall",
                            "prompt": "🧠 Say 'quotient 8' so you keep the result ready.",
                            "memory_hook": "Hold the quotient 8 for the next step.",
                            "visual": "🧠➡️8",
                            "parallel_example": "Say 'quotient 4' for 12 ÷ 3 👍"
                        },
                        {
                            "label": "🎮 Example",
                            "prompt": "🎮 Solve 12 ÷ 3 = 4 with mini bricks.",
                            "memory_hook": "Notice how the simpler split mirrors this division.",
                            "visual": "12 ÷ 3 → 4",
                            "parallel_example": "12 ÷ 3 = 4 using 🧱🧱🧱🧱."
                        }
                    ],
                    "bridge": {
                        "prompt": "What is 12 ÷ 3 and what does it represent?",
                        "hint": "Try: share 12 blocks into three equal lines.",
                        "success_keywords": ["divide", "12", "3"],
                    },
                },
                {
                    "id": "step-3",
                    "title": "Add The Result",
                    "objective": "Add the quotient to 5 to finish the expression.",
                    "check_prompt": "What do you get when you add the quotient to 5?",
                    "success_keywords": ["add", "8", "5", "sum"],
                    "hint": "Example: add 8 and 5 using quick mental math.",
                    "substeps": [
                        {
                            "label": "🔗 Link",
                            "prompt": "🔗 Connect the quotient 8 with the +5 you spotted earlier.",
                            "memory_hook": "Grab the 8 you said out loud in Step 2.",
                            "visual": "8 ➕ 5",
                            "parallel_example": "Link 4 with +2 to make 6 😀"
                        },
                        {
                            "label": "🧠 Recall",
                            "prompt": "🧠 Whisper '8 plus 5' before writing anything down.",
                            "memory_hook": "Keep repeating the quotient while you add.",
                            "visual": "🧠➕",
                            "parallel_example": "Chant '4 plus 2' 🎵"
                        },
                        {
                            "label": "🎨 Sketch",
                            "prompt": "🎨 Draw eight stars, add five more, then count them.",
                            "memory_hook": "Point to the stars as you count so you stay accurate.",
                            "visual": "⭐⭐⭐⭐⭐⭐⭐⭐ + ⭐⭐⭐⭐⭐",
                            "parallel_example": "Sketch four stars + two stars ⭐⭐⭐⭐ + ⭐⭐"
                        }
                    ],
                    "bridge": {
                        "prompt": "Add 4 + 2 after finding 12 ÷ 3 = 4.",
                        "hint": "Try: reuse the quotient before adding the new number.",
                        "success_keywords": ["add", "quotient", "4"],
                    },
                },
                {
                    "id": "step-4",
                    "title": "Check Your Work",
                    "objective": "Verify the order makes sense and say the final answer out loud.",
                    "check_prompt": "What is the final answer to 48 ÷ 6 + 5 and why does it make sense?",
                    "success_keywords": ["answer", "13", "because"],
                    "hint": "Example: say 'It equals 13 because I divided first, then added.'",
                    "substeps": [
                        {
                            "label": "🔁 Replay",
                            "prompt": "🔁 Repeat: divide 48 by 6, then add 5 again.",
                            "memory_hook": "Compare each replay with Steps 2 and 3.",
                            "visual": "48 ÷ 6 → 8 → +5",
                            "parallel_example": "Redo 12 ÷ 3 + 2 → 6 ✅"
                        },
                        {
                            "label": "🧠 Recall",
                            "prompt": "🧠 Say 'I got 8, then 13' to lock in the results.",
                            "memory_hook": "Hold both the quotient and the final sum together.",
                            "visual": "🧠 8 & 13",
                            "parallel_example": "Keep '4 then 6' in mind 🧠"
                        },
                        {
                            "label": "🎯 Sketch",
                            "prompt": "🎯 Draw a number line jumping to 8, then leaping to 13.",
                            "memory_hook": "Use the jumps to argue the order of operations.",
                            "visual": "0 → 8 → 13",
                            "parallel_example": "Hop 0 → 4 → 6 on a mini line 📈"
                        }
                    ],
                    "bridge": {
                        "prompt": "Share the answer to 12 ÷ 3 + 2 and explain why it's correct.",
                        "hint": "Example: say 'It equals 6 since I divided then added 2.'",
                        "success_keywords": ["answer", "6", "because"],
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


_OPERATION_REGEX: dict[str, re.Pattern[str]] = {
    "addition": re.compile(
        r"(?:\badd(?:ing|ed)?\b|\bplus\b|\bsum\b|\btotal\b|\ball together\b|\bcombined\b|\bincrease\b|\+)",
        re.IGNORECASE,
    ),
    "subtraction": re.compile(
        r"(?:\bsubtract(?:ing|ed)?\b|\bminus\b|\bdifference\b|\btake away\b|\bafter\b|\bsold\b|\bspent\b|"
        r"\bleft(?:over)?\b|\bremain(?:ing)?\b|\blost\b|\-)",
        re.IGNORECASE,
    ),
    "multiplication": re.compile(
        r"(?:\bmultiply(?:ing|ied)?\b|\btimes\b|\bproduct\b|\bper\b\s+\w+|\bfor each\b|\beach\b\s+\w+\s+(?:gets|has|hold(?:s)?)|"
        r"\bgroups?\s+of\b|\bbox(?:es)?\s+of\b|\bsets?\s+of\b|\bpack(?:s)?\s+of\b|\*)",
        re.IGNORECASE,
    ),
    "division": re.compile(
        r"(?:\bdivide(?:d|s|ing)?\b|\bquotient\b|\bsplit\b|\bshare\b|\bequally\b|\bper\b|\beach\b|\binto\b\s+\d+\s+\w+|/)",
        re.IGNORECASE,
    ),
    "fraction": re.compile(r"(?:\bfraction\b|\bnumerator\b|\bdenominator\b|\bover\b)", re.IGNORECASE),
    "percent": re.compile(r"(?:\bpercent(?:age)?\b|%)", re.IGNORECASE),
    "equation": re.compile(r"(?:\bequation\b|\bsolve for\b|\bvariable\b|\bunknown\b|=)", re.IGNORECASE),
    "exponent": re.compile(r"(?:\bpower\b|\bsquared\b|\bcubed\b|\^)", re.IGNORECASE),
}

_REMOVAL_KEYWORDS = [
    "sold",
    "sell",
    "gave",
    "give away",
    "gave away",
    "left",
    "leftover",
    "remaining",
    "remain",
    "remains",
    "spent",
    "used",
    "lost",
]
_SHARE_KEYWORDS = ["each", "per", "equally", "share", "split", "every"]
_CONTAINER_KEYWORDS = ["box", "boxes", "basket", "baskets", "bag", "bags", "pack", "packs", "container", "containers"]

_NUMBER_UNIT_STOPWORDS = {"the", "a", "an", "each", "per", "of", "to", "into", "by", "with", "for", "in", "on"}
_NUMBER_UNIT_REGEX = re.compile(r"\b(\d+(?:\.\d+)?)\s*([a-zA-Z][a-zA-Z\-]*)?")

_EMOJI_REGEX = re.compile(r"[\u2190-\u21FF\u2600-\u27BF\u2B00-\u2BFF\u1F300-\u1FAFF]")


_STEP_TEMPLATES: dict[str, dict[str, Any]] = {
    "understand": {
        "title": "Understand Problem",
        "objective": "Restate the question about {context} and list the important numbers.",
        "check_prompt": "What information do you know and what is the question asking you to find?",
        "hint": "Example: list the key numbers and what they represent.",
        "keywords": ["understand", "numbers", "question"],
        "focus_tip": "🧩 Break the problem into the numbers and operations you spotted.",
        "example": "Circle 20 ÷ 5 + 4 with doodles ✏️✏️",
        "visual": "📝 20 ÷ 5 + 4",
        "bridge_prompt": "List the numbers in 20 ÷ 5 + 4 and explain each one.",
        "bridge_hint": "Try: say each number and symbol aloud.",
        "bridge_keywords": ["20", "5", "4"],
        "allow_suffix": False,
    },
    "strategy": {
        "title": "Plan Strategy",
        "objective": "Decide which operation fits first and why it matches {context}.",
        "check_prompt": "Which operation comes first and what clue word convinces you?",
        "hint": "Example: pick division if you see 'each' or 'equally' in the problem.",
        "keywords": ["plan", "order", "strategy"],
        "focus_tip": "🧩 Highlight clue words like 'each', 'total', or 'left' before you choose the first move.",
        "example": "Circle 'equally' then write 'divide first' on your plan card.",
        "visual": "🧭 divide → add",
        "bridge_prompt": "For 12 ÷ 3 + 2, which operation would you do first and why?",
        "bridge_hint": "Try: say 'Divide first because of the ÷ sign.'",
        "bridge_keywords": ["order", "divide", "add"],
        "allow_suffix": False,
    },
    "practice": {
        "title": "Micro Example",
        "objective": "Test the plan with a tiny practice example before the real numbers.",
        "check_prompt": "What did your mini example teach you for the main problem?",
        "hint": "Example: swap in small numbers to rehearse the move.",
        "keywords": ["practice", "example", "mini"],
        "focus_tip": "🧩 Swap the big numbers for tiny ones and try the move.",
        "example": "Play with 2 + 1 = 3 using triangles 🔺🔺🔺",
        "visual": "🔺➕🔺 = 3",
        "bridge_prompt": "Explain the pattern you see in 2 + 1 = 3.",
        "bridge_hint": "Try: say 'two plus one equals three' aloud.",
        "bridge_keywords": ["two", "plus", "three"],
        "allow_suffix": True,
    },
    "addition": {
        "title": "Add The Pieces",
        "objective": "Combine your last result with {third_number_phrase} to get the total {final_unit}.",
        "check_prompt": "What total of {final_unit} do you get after adding the new amount?",
        "hint": "Example: line up your previous result with the new amount, then add them.",
        "keywords": ["add", "sum", "total"],
        "focus_tip": "🧩 Pull the answer from {previous_step} next to {third_number_phrase} before you add.",
        "example": "Add 8 cupcakes + 5 cupcakes with star doodles ⭐⭐⭐⭐⭐⭐",
        "visual": "⭐➕⭐⭐⭐",
        "bridge_prompt": "Add 4 + 2 after repeating the first number aloud.",
        "bridge_hint": "Try: say 'four plus two' before writing the total.",
        "bridge_keywords": ["add", "total", "six"],
        "allow_suffix": True,
    },
    "subtraction": {
        "title": "Subtract Amount",
        "objective": "Subtract the {removal_verb} amount from {first_number_phrase} to see what remains.",
        "check_prompt": "How many {final_unit} remain after subtracting the amount you just found? Share the final answer.",
        "hint": "Example: write {first_number_phrase} minus the amount removed, then state what is left.",
        "keywords": ["subtract", "difference", "minus"],
        "focus_tip": "🧩 Line up {first_number_phrase} above the amount you {removal_verb} and cross out what goes away.",
        "example": "120 - 45 = 75 with cupcake doodles 🧁🧁🧁",
        "visual": "⚪⚪⚪⚪⚪",
        "bridge_prompt": "Try 9 - 4 using counters and say how many are left.",
        "bridge_hint": "Try: cross out four dots from nine before counting.",
        "bridge_keywords": ["left", "difference", "five"],
        "allow_suffix": True,
    },
    "multiplication": {
        "title": "Multiply Groups",
        "objective": "Multiply the per-{container_unit} amount from {previous_step} by {multiplier_phrase}.",
        "check_prompt": "How many {first_unit} are involved after multiplying by {multiplier_phrase}?",
        "hint": "Example: copy the per-{container_unit} count and multiply by the number of {container_unit}.",
        "keywords": ["multiply", "product", "times"],
        "focus_tip": "🧩 Sketch {multiplier_phrase} groups, each holding the amount from {previous_step}.",
        "example": "If 1 box holds 15 cupcakes, draw 3 boxes of 15 🧁🧁🧁.",
        "visual": "⭐⭐ x3",
        "bridge_prompt": "Multiply 4 cupcakes per basket by 2 baskets.",
        "bridge_hint": "Try: draw two baskets with four cupcakes each.",
        "bridge_keywords": ["multiply", "baskets", "total"],
        "allow_suffix": True,
    },
    "division": {
        "title": "Divide Each {container_unit_title}",
        "objective": "Divide {first_number_phrase} by {divisor_phrase} to learn how many {first_unit} sit in one {container_unit_singular}.",
        "check_prompt": "How many {first_unit} fit in each {container_unit_singular}? Explain the clue that told you to divide.",
        "hint": "Example: split {first_number_phrase} equally among {divisor_phrase} before counting a single group.",
        "keywords": ["divide", "quotient", "share"],
        "focus_tip": "🧩 Use counters to split {first_number_phrase} into {divisor_phrase} equal piles before counting one.",
        "example": "Share 12 apples into 3 baskets 🍎🍎🍎.",
        "visual": "🧺🧺🧺",
        "bridge_prompt": "Divide 12 by 3 to see how many go in each group.",
        "bridge_hint": "Try: hand out 12 blocks into three rows evenly.",
        "bridge_keywords": ["divide", "each", "share"],
        "allow_suffix": True,
    },
    "fraction": {
        "title": "Model Fractions",
        "objective": "Represent the fraction pieces before computing with them.",
        "check_prompt": "How many parts are shaded and what fraction does that show?",
        "hint": "Example: draw the fraction with equal slices first.",
        "keywords": ["fraction", "parts", "shade"],
        "focus_tip": "🧩 Split a shape into equal pieces and mark the needed ones.",
        "example": "Shade 1/4 of 8 squares ▢▢▢▢",
        "visual": "🟦▢▢▢",
        "bridge_prompt": "Shade 1/2 of 6 squares and name the fraction.",
        "bridge_hint": "Try: color three out of six boxes.",
        "bridge_keywords": ["fraction", "shade", "half"],
        "allow_suffix": True,
    },
    "percent": {
        "title": "Work With Percents",
        "objective": "Convert the percent to a friendlier number before using it.",
        "check_prompt": "What number represents the percent in this problem?",
        "hint": "Example: turn the percent into a fraction or decimal first.",
        "keywords": ["percent", "decimal", "fraction"],
        "focus_tip": "🧩 Turn the percent into a fraction you can use quickly.",
        "example": "Find 10% of 50 using ten dots ⚫⚫⚫⚫⚫⚫⚫⚫⚫⚫",
        "visual": "⚫⚫⚫⚫⚫",
        "bridge_prompt": "Find 10% of 30 and describe the steps.",
        "bridge_hint": "Try: move the decimal one place left.",
        "bridge_keywords": ["percent", "10", "30"],
        "allow_suffix": True,
    },
    "equation": {
        "title": "Solve Equation",
        "objective": "Isolate the variable using the inverse operation.",
        "check_prompt": "Which inverse operation solves the equation and what value do you get?",
        "hint": "Example: move terms with inverse operations on both sides.",
        "keywords": ["solve", "equation", "variable"],
        "focus_tip": "🧩 Undo one operation at a time to isolate the variable.",
        "example": "Solve x + 3 = 9 by subtracting 3 ➡️",
        "visual": "🧮 x + 3 = 9",
        "bridge_prompt": "Solve y + 2 = 5 and explain the step.",
        "bridge_hint": "Try: subtract two from both sides.",
        "bridge_keywords": ["solve", "2", "5"],
        "allow_suffix": True,
    },
    "exponent": {
        "title": "Tackle Exponents",
        "objective": "Evaluate the power before the other operations.",
        "check_prompt": "What value do you get after handling the exponent?",
        "hint": "Example: multiply the base by itself as many times as needed.",
        "keywords": ["power", "exponent", "square"],
        "focus_tip": "🧩 Expand the repeated multiplication for the exponent.",
        "example": "Compute 2^3 = 8 using cubes 🧊🧊🧊🧊🧊🧊🧊🧊",
        "visual": "🧊 2^3 = 8",
        "bridge_prompt": "Find 3^2 and say what it means.",
        "bridge_hint": "Try: multiply three by itself.",
        "bridge_keywords": ["exponent", "3", "2"],
        "allow_suffix": True,
    },
    "check": {
        "title": "Check Your Work",
        "objective": "Replay the steps and confirm the final answer makes sense.",
        "check_prompt": "Restate the final answer for {context} and explain how you know it is correct.",
        "hint": "Example: redo each step quickly, then say the final answer with units.",
        "keywords": ["check", "verify", "order"],
        "focus_tip": "🧩 Replay each step and look for mismatches with {previous_step}.",
        "example": "Redo 12 ÷ 3 + 2 to confirm 6 ✅",
        "visual": "🔁 divide → add",
        "bridge_prompt": "Check 12 ÷ 3 + 2 on a number line.",
        "bridge_hint": "Try: hop four, then +2 to land on six.",
        "bridge_keywords": ["check", "6", "line"],
        "allow_suffix": False,
    },
    "result": {
        "title": "Share Final Answer",
        "objective": "State the final answer in {final_unit} and tie it back to the story.",
        "check_prompt": "What is the final answer for {context}? Say the number and {final_unit}.",
        "hint": "Example: say the answer with units and a quick because statement.",
        "keywords": ["answer", "final", "result"],
        "focus_tip": "🧩 Point to your previous step before announcing the answer out loud.",
        "example": "Say 'There are 13 pieces left' with a big checkmark ✅.",
        "visual": "✅ final answer",
        "bridge_prompt": "Share the answer for the mini example and explain why it fits.",
        "bridge_hint": "Try: say the mini answer plus a 'because' sentence.",
        "bridge_keywords": ["answer", "because"],
        "allow_suffix": False,
    },
}

_KIND_KEYWORDS: dict[str, list[str]] = {
    "addition": ["add", "sum", "plus"],
    "subtraction": ["subtract", "minus", "difference"],
    "multiplication": ["multiply", "product", "times"],
    "division": ["divide", "quotient", "share"],
    "fraction": ["fraction", "numerator", "denominator"],
    "percent": ["percent", "percentage", "percent"],
    "equation": ["equation", "variable", "solve"],
    "exponent": ["power", "exponent", "square"],
    "strategy": ["plan", "strategy", "order"],
    "check": ["check", "verify", "review"],
    "practice": ["example", "practice", "mini"],
    "result": ["answer", "final", "result"],
}


def _shorten_text(value: str, limit: int = 80) -> str:
    trimmed = (value or "").strip()
    if len(trimmed) <= limit:
        return trimmed
    return trimmed[: limit - 3].rstrip() + "..."


def _format_template(template: str, values: dict[str, str]) -> str:
    try:
        return template.format(**values)
    except (KeyError, IndexError):
        return template


def _singularize(word: str) -> str:
    if not word:
        return "group"
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith(("ses", "xes", "zes", "ches", "shes")):
        return word[:-2]
    if word.endswith("es") and len(word) > 2:
        return word[:-2]
    if word.endswith("s") and len(word) > 1:
        return word[:-1]
    return word


def _build_format_values(assignment_request: str) -> dict[str, str]:
    context = assignment_request.strip() or "this assignment"
    matches = list(_NUMBER_UNIT_REGEX.finditer(assignment_request))
    numbers: list[str] = []
    units: list[str] = []
    for match in matches:
        number = match.group(1)
        unit = (match.group(2) or "").strip(".,!? ").lower()
        numbers.append(number)
        if unit in _NUMBER_UNIT_STOPWORDS:
            unit = ""
        units.append(unit)
    while len(units) < len(numbers):
        units.append("")

    def _resolve_unit(index: int, fallback: str) -> str:
        if 0 <= index < len(units) and units[index]:
            return units[index]
        return fallback

    first_unit = _resolve_unit(0, "items")
    second_unit = _resolve_unit(1, "groups" if first_unit != "groups" else "piles")
    third_unit = _resolve_unit(2, second_unit if second_unit != "groups" else first_unit)
    final_unit = first_unit or "items"

    lowered = assignment_request.lower()
    container_unit = next((word for word in _CONTAINER_KEYWORDS if word in lowered), second_unit or "groups")
    container_unit_singular = _singularize(container_unit)
    container_unit_title = container_unit_singular.title() if container_unit_singular else "Group"
    question_unit_match = re.search(r"how many\s+([a-zA-Z\-]+)", lowered)
    question_unit = question_unit_match.group(1) if question_unit_match else final_unit
    removal_verb = next((word for word in _REMOVAL_KEYWORDS if word in lowered), "remove")

    numbers_joined = ", ".join(numbers[:3]) if numbers else "the numbers"

    def _number_phrase(index: int, unit: str, fallback: str) -> str:
        if index < len(numbers):
            if unit:
                return f"{numbers[index]} {unit}"
            return numbers[index]
        return fallback

    divisor_phrase = _number_phrase(1, second_unit, "the number of groups")
    multiplier_phrase = (
        _number_phrase(2, container_unit, "how many groups are involved")
        if len(numbers) > 2
        else _number_phrase(1, container_unit, "how many groups are involved")
    )
    remaining_phrase = f"{question_unit} remaining"

    return {
        "context": _shorten_text(context, 90),
        "numbers": numbers_joined,
        "first_number": numbers[0] if numbers else "",
        "second_number": numbers[1] if len(numbers) > 1 else "",
        "third_number": numbers[2] if len(numbers) > 2 else "",
        "first_number_or_word": numbers[0] if numbers else "the first number",
        "second_number_or_word": numbers[1] if len(numbers) > 1 else "the next number",
        "third_number_or_word": numbers[2] if len(numbers) > 2 else "a new number",
        "first_unit": first_unit,
        "second_unit": second_unit,
        "third_unit": third_unit,
        "final_unit": question_unit or final_unit,
        "container_unit": container_unit,
        "container_unit_singular": container_unit_singular or "group",
        "container_unit_title": container_unit_title,
        "removal_verb": removal_verb,
        "first_number_phrase": _number_phrase(0, first_unit, "the starting amount"),
        "second_number_phrase": _number_phrase(1, second_unit, "the second amount"),
        "third_number_phrase": _number_phrase(2, third_unit, "the extra number"),
        "question_unit": question_unit or final_unit,
        "divisor_phrase": divisor_phrase,
        "multiplier_phrase": multiplier_phrase,
        "remaining_phrase": remaining_phrase,
    }


def _generate_substeps_for(
    title: str,
    focus_tip_template: str,
    example: str,
    visual: str,
    index: int,
    previous_title: str | None,
) -> list[TutorSubstep]:
    previous_reference = (
        previous_title
        if previous_title
        else ("the problem question" if index == 1 else "the previous step")
    )
    focus_prompt = focus_tip_template.replace("{previous_step}", previous_reference)
    recall_prompt = (
        f"🧠 Say the result from {previous_reference} before you continue {title.lower()}."
        if index > 1 or previous_title
        else "🧠 Repeat the exact question so it sticks before you compute."
    )
    recall_hook = (
        f"Reuse what you found in {previous_reference}."
        if index > 1 or previous_title
        else "Keep the original question in mind for the next move."
    )
    doodle_prompt = f"🎨 Sketch the mini example: {example}"
    resolved_visual = visual or "🎨✨"
    return [
        TutorSubstep(
            label="🧩 Break",
            prompt=focus_prompt,
            memory_hook=f"Glance back at {previous_reference} as you start.",
            visual=resolved_visual,
            parallel_example=example,
        ),
        TutorSubstep(
            label="🧠 Recall",
            prompt=recall_prompt,
            memory_hook=recall_hook,
            visual="🧠➡️✍️",
            parallel_example=example,
        ),
        TutorSubstep(
            label="🎨 Sketch",
            prompt=doodle_prompt,
            memory_hook="Keep the doodle visible while you work.",
            visual=resolved_visual,
            parallel_example=example,
        ),
    ]


def _detect_operations(assignment_request: str, max_ops: int | None = None) -> list[str]:
    matches: list[tuple[int, str]] = []
    for name, pattern in _OPERATION_REGEX.items():
        for match in pattern.finditer(assignment_request):
            matches.append((match.start(), name))
    matches = _post_process_operations(assignment_request, matches)
    matches.sort(key=lambda item: item[0])
    filtered: list[tuple[int, str]] = []
    last_pos_by_name: dict[str, int] = {}
    for position, name in matches:
        last_position = last_pos_by_name.get(name)
        if last_position is None or position - last_position > 30:
            filtered.append((position, name))
            last_pos_by_name[name] = position
    ordered: list[str] = [name for _, name in filtered]
    if max_ops is not None:
        return ordered[:max_ops]
    return ordered


def _post_process_operations(
    assignment_request: str, matches: list[tuple[int, str]]
) -> list[tuple[int, str]]:
    lowered = assignment_request.lower()
    existing_names = [name for _, name in matches]

    def add_operation(name: str, keyword: str) -> None:
        if name in existing_names:
            return
        position = lowered.find(keyword)
        if position < 0:
            position = len(lowered) + len(matches)
        matches.append((position, name))
        existing_names.append(name)

    if any(keyword in lowered for keyword in _SHARE_KEYWORDS):
        add_operation("division", next((kw for kw in _SHARE_KEYWORDS if kw in lowered), "divide"))

    if any(keyword in lowered for keyword in _REMOVAL_KEYWORDS):
        add_operation("subtraction", next((kw for kw in _REMOVAL_KEYWORDS if kw in lowered), "subtract"))

    if any(word in lowered for word in ["total", "altogether", "combined"]) and "addition" not in existing_names:
        add_operation("addition", "total")

    if any(keyword in lowered for keyword in ["twice", "double", "triple"]):
        add_operation("multiplication", next((kw for kw in ["twice", "double", "triple"] if kw in lowered), "multiply"))

    sold_index = next(
        (lowered.find(keyword) for keyword in ["sold", "sell", "gave", "spent", "used"] if keyword in lowered), -1
    )
    if sold_index != -1 and any(container in lowered for container in _CONTAINER_KEYWORDS):
        if "subtraction" not in existing_names:
            matches.append((sold_index + 0.1, "subtraction"))
            existing_names.append("subtraction")
        if "multiplication" not in existing_names:
            matches.append((sold_index + 0.2, "multiplication"))
            existing_names.append("multiplication")

    return matches


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
        prompt=f"🎮 Mini version: redo {step.title.lower()} with numbers 3 and 2.",
        hint="Try: play the tiny example first, then return.",
        success_keywords=_extract_keywords(keywords_seed, limit=2) or ["mini", "example"],
    )


def _ensure_visual_text(text: str | None, fallback: str) -> str:
    stripped = (text or "").strip()
    if not stripped:
        return fallback
    if not _EMOJI_REGEX.search(stripped):
        return f"{stripped} ✏️"
    return stripped


def _build_step_from_spec(
    idx: int,
    spec_key: str,
    format_values: dict[str, str],
    previous_title: str | None,
    occurrence_index: int = 1,
    operation_name: str | None = None,
) -> TutorGameStep:
    spec = _STEP_TEMPLATES.get(spec_key, _STEP_TEMPLATES["practice"])
    call_values = dict(format_values)
    call_values.update(
        {
            "operation_name": spec_key,
            "occurrence": str(occurrence_index),
            "previous_step": previous_title or "the previous step",
            "operation_word": operation_name or format_values.get("primary_operation") or "the first move",
        }
    )
    title_template = spec["title"]
    title = _format_template(title_template, call_values)
    if spec.get("allow_suffix", False) and occurrence_index > 1:
        title = f"{title} {occurrence_index}"
    objective = _format_template(spec["objective"], call_values)
    check_prompt = _format_template(spec["check_prompt"], call_values)
    hint = _format_template(spec["hint"], call_values)
    focus_tip_template = _format_template(spec["focus_tip"], call_values)
    example = _format_template(spec["example"], call_values)
    visual = _format_template(spec["visual"], call_values)
    bridge_prompt = _format_template(spec["bridge_prompt"], call_values)
    bridge_hint = _format_template(spec["bridge_hint"], call_values)
    success_keywords = _ensure_keywords(spec.get("keywords", []), f"{objective} {check_prompt}")
    if operation_name:
        success_keywords = _ensure_keywords(success_keywords + [operation_name], f"{objective} {check_prompt}")
    bridge_keywords = _ensure_keywords(
        spec.get("bridge_keywords", []),
        bridge_prompt,
        fallback_token="clue",
    )
    substeps = _generate_substeps_for(
        title=title,
        focus_tip_template=focus_tip_template,
        example=example,
        visual=visual,
        index=idx,
        previous_title=previous_title,
    )
    return TutorGameStep(
        id=f"step-{idx}",
        title=title,
        objective=objective,
        check_prompt=check_prompt,
        success_keywords=success_keywords,
        hint=hint,
        bridge=TutorBridgeStep(
            prompt=bridge_prompt,
            hint=bridge_hint,
            success_keywords=bridge_keywords,
        ),
        substeps=substeps,
    )


def _infer_step_kind(step: TutorGameStep) -> str:
    text = f"{step.title} {step.objective} {step.check_prompt}".lower()
    for kind, tokens in _KIND_KEYWORDS.items():
        if any(token in text for token in tokens):
            return kind
    return "practice"


def _ensure_substeps(step: TutorGameStep, index: int, previous_title: str | None) -> list[TutorSubstep]:
    existing = [substep for substep in step.substeps if substep and substep.prompt]
    inferred_kind = _infer_step_kind(step)
    template = _STEP_TEMPLATES.get(inferred_kind, _STEP_TEMPLATES["practice"])
    fallback_substeps = _generate_substeps_for(
        title=step.title or template["title"],
        focus_tip_template=template["focus_tip"],
        example=template["example"],
        visual=template["visual"],
        index=index,
        previous_title=previous_title,
    )
    sanitized: list[TutorSubstep] = []
    for sub_index in range(3):
        fallback = fallback_substeps[sub_index] if sub_index < len(fallback_substeps) else fallback_substeps[-1]
        if sub_index < len(existing):
            candidate = existing[sub_index]
            updates: dict[str, Any] = {}
            if not candidate.label:
                updates["label"] = fallback.label
            prompt = _ensure_visual_text(candidate.prompt, fallback.prompt)
            if prompt != candidate.prompt:
                updates["prompt"] = prompt
            if not candidate.memory_hook:
                updates["memory_hook"] = fallback.memory_hook
            if candidate.visual:
                visual = _ensure_visual_text(candidate.visual, fallback.visual)
                if visual != candidate.visual:
                    updates["visual"] = visual
            else:
                updates["visual"] = fallback.visual
            if not candidate.parallel_example:
                updates["parallel_example"] = fallback.parallel_example
            candidate = candidate.model_copy(update=updates) if updates else candidate
        else:
            candidate = fallback
        sanitized.append(candidate)
    return sanitized


def _ensure_final_answer_details(step: TutorGameStep, format_values: dict[str, str]) -> TutorGameStep:
    final_unit = format_values.get("final_unit", "items")
    context = format_values.get("context", "this problem")
    prompt = (step.check_prompt or "").strip()
    if "final answer" not in prompt.lower():
        prompt_core = prompt.rstrip("?")
        if prompt_core:
            new_prompt = f"{prompt_core}? What is the final answer for {context}? Say it with {final_unit}."
        else:
            new_prompt = f"What is the final answer for {context}? Say it with {final_unit}."
    else:
        new_prompt = prompt
    hint = (step.hint or "").strip()
    if hint and not hint.lower().startswith(("example:", "try:")):
        hint = f"Example: {hint.lstrip()}"
    if "final answer" not in hint.lower():
        hint = f"Example: say the final answer with {final_unit} and explain why it fits."
    updated_keywords = _ensure_keywords(step.success_keywords, new_prompt, fallback_token="answer")
    updated_substeps = list(step.substeps or [])
    if updated_substeps:
        first_substep = updated_substeps[0]
        first_prompt = (first_substep.prompt or "").strip()
        if "final answer" not in first_prompt.lower():
            appended_prompt = f"{first_prompt.rstrip('.')} → say the final answer with {final_unit}."
            updated_substeps[0] = first_substep.model_copy(
                update={"prompt": _ensure_visual_text(appended_prompt, appended_prompt)}
            )
    return step.model_copy(
        update={
            "check_prompt": new_prompt,
            "hint": hint,
            "success_keywords": updated_keywords,
            "substeps": updated_substeps,
        }
    )


def _safe_float(token: str | None) -> float | None:
    if not token:
        return None
    try:
        return float(token)
    except (TypeError, ValueError):
        return None


def _format_number(value: float | None) -> str:
    if value is None:
        return ""
    rounded = round(value)
    if abs(value - rounded) < 1e-6:
        return str(int(rounded))
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return text


def _build_two_step_removal_plan(format_values: dict[str, str]) -> TutorGamePlan:
    context = format_values.get("context", "this assignment")
    first_number = format_values.get("first_number")
    second_number = format_values.get("second_number")
    third_number = format_values.get("third_number")
    first_unit = format_values.get("first_unit", "items")
    final_unit = format_values.get("final_unit", first_unit)
    container_unit = format_values.get("container_unit_singular", "group")
    divisor_phrase = format_values.get("divisor_phrase", "the groups")

    dividend = _safe_float(first_number)
    divisor = _safe_float(second_number)
    removed = _safe_float(third_number)
    per_container = dividend / divisor if dividend is not None and divisor not in (None, 0) else None
    boxes_left = divisor - removed if divisor is not None and removed is not None else None
    remaining = per_container * boxes_left if per_container is not None and boxes_left is not None else None

    per_container_str = _format_number(per_container) or "the per-box amount"
    boxes_left_str = _format_number(boxes_left) or "boxes left"
    remaining_str = _format_number(remaining) or "the total"

    step1 = TutorGameStep(
        id="step-1",
        title=f"Divide Per {container_unit.title()}",
        objective=f"Find how many {first_unit} go in each {container_unit}.",
        check_prompt=f"What is {first_number} ÷ {second_number}? How many {first_unit} per {container_unit}?",
        success_keywords=_ensure_keywords(
            [
                first_number or "",
                second_number or "",
                "divide",
                "quotient",
                per_container_str,
                container_unit,
            ],
            f"{first_number} divided by {second_number}",
            fallback_token="divide",
        ),
        hint=f"Example: divide {first_number} by {second_number} so each {container_unit} gets the same share.",
        bridge=TutorBridgeStep(
            prompt=f"Divide 12 cupcakes into 3 {container_unit}s. How many per {container_unit}?",
            hint="Try: give one cupcake to each box until you run out.",
            success_keywords=["12", "3", "4"],
        ),
        substeps=[
            TutorSubstep(
                label="🍰 Share",
                prompt=f"🍰 Split {format_values['first_number_phrase']} equally into {divisor_phrase}.",
                memory_hook="Remember the per-box amount for later.",
                visual=f"{first_number or ''} ÷ {second_number or ''}",
                parallel_example="Share 12 cupcakes into 3 boxes 🧁🧁🧁.",
            ),
            TutorSubstep(
                label="🧠 Recall",
                prompt=f"🧠 Say 'Each {container_unit} gets ___ {first_unit}' to store the result.",
                memory_hook="Keep this number handy for the next move.",
                visual="🧠→🍰",
                parallel_example="Whisper 'Each box gets 4 cupcakes.'",
            ),
            TutorSubstep(
                label="🎯 Practice",
                prompt="🎯 Try a mini version: 12 ÷ 3 = 4 cupcakes per box.",
                memory_hook="Notice how equal sharing gives a quotient.",
                visual="12 ÷ 3 = 4",
                parallel_example="12 cupcakes ➗ 3 boxes = 4 cupcakes each.",
            ),
        ],
    )

    step2_raw = TutorGameStep(
        id="step-2",
        title="Combine To Finish",
        objective=f"Use boxes left and cupcakes per {container_unit} to find the remaining {final_unit}.",
        check_prompt=(
            f"After selling {third_number} {container_unit}s, how many {final_unit} remain? "
            f"Use boxes left × cupcakes per {container_unit}."
        ),
        success_keywords=_ensure_keywords(
            [
                boxes_left_str,
                per_container_str,
                remaining_str,
                final_unit,
                "left",
                "remain",
            ],
            f"{boxes_left_str} times {per_container_str}",
            fallback_token="answer",
        ),
        hint=(
            f"Example: ({second_number} − {third_number}) × {per_container_str} gives the remaining {final_unit}."
        ),
        bridge=TutorBridgeStep(
            prompt="If 20 cupcakes fill 4 boxes and 1 box is sold, how many cupcakes are left?",
            hint="Try: 4 − 1 boxes = 3, then 3 × 5 cupcakes = 15.",
            success_keywords=["15", "3", "5"],
        ),
        substeps=[
            TutorSubstep(
                label="➖ Boxes",
                prompt=f"➖ Subtract: {second_number} − {third_number} to see boxes left.",
                memory_hook="We only keep the boxes still in the bakery.",
                visual=f"{second_number or ''} - {third_number or ''}",
                parallel_example="6 boxes − 2 sold = 4 boxes left 📦",
            ),
            TutorSubstep(
                label="🔁 Reuse",
                prompt=f"🔁 Grab the per-box amount ({per_container_str}) from Step 1.",
                memory_hook="Use this value when you multiply.",
                visual=f"🧠 → {per_container_str}",
                parallel_example="Hold onto '4 cupcakes per box' from the practice run.",
            ),
            TutorSubstep(
                label="✖️ Multiply",
                prompt=f"✖️ Multiply boxes left ({boxes_left_str}) by cupcakes per box ({per_container_str}).",
                memory_hook="This gives the remaining {final_unit}.",
                visual=f"{boxes_left_str} × {per_container_str}",
                parallel_example="4 boxes × 4 cupcakes = 16 cupcakes left.",
            ),
        ],
    )

    step2 = _ensure_final_answer_details(step2_raw, format_values)
    overview = (
        f"We'll beat the challenge in two moves: divide per {container_unit} and multiply the boxes left."
    )
    return TutorGamePlan(assignment_overview=overview, steps=[step1, step2])


def _estimate_plan_length(assignment_request: str, operations: Sequence[str] | None = None) -> int:
    operations_list = list(operations) if operations is not None else _detect_operations(assignment_request)
    op_count = len(operations_list)
    if op_count == 0:
        base_steps = 3
    elif op_count == 1:
        base_steps = 3
    elif op_count == 2:
        base_steps = 4
    else:
        base_steps = min(5, op_count)
    text = assignment_request.lower()
    number_tokens = len(re.findall(r"\d+(?:\.\d+)?", text))
    operation_tokens = op_count
    sentence_markers = text.count(".") + text.count("?") + text.count("!")
    clause_markers = len(re.findall(r"(and then|after|before|together|each|per)", text))
    multi_step_markers = sum(
        1
        for keyword in [
            "multi-step",
            "multi step",
            "two-part",
            "word problem",
            "equation system",
            "fractions",
            "percent",
        ]
        if keyword in text
    )
    length_bonus = 1 if len(text) > 120 else 0
    complexity_score = (
        number_tokens + operation_tokens + sentence_markers + clause_markers + multi_step_markers + length_bonus
    )
    bonus = 1 if (complexity_score >= 6 and base_steps < 5 and op_count <= 2) else 0
    target = min(5, base_steps + bonus)
    target = max(target, 3)
    target = max(target, op_count if op_count <= 5 else 5)
    return target


def normalize_tutor_plan(plan: TutorGamePlan, assignment_request: str) -> TutorGamePlan:
    format_values = _build_format_values(assignment_request)
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
        bridge_hint = normalized_bridge.hint or ""
        if bridge_hint and not bridge_hint.lower().startswith(("example:", "try:")):
            normalized_bridge = normalized_bridge.model_copy(update={"hint": f"Try: {bridge_hint.lstrip()}"})
        previous_title = normalized_steps[-1].title if normalized_steps else None
        provisional_step = step.model_copy(
            update={
                "id": step.id or f"step-{index}",
                "title": step.title or f"Checkpoint {index}",
                "success_keywords": _ensure_keywords(step.success_keywords, keywords_source),
                "bridge": normalized_bridge,
            }
        )
        hint_text = provisional_step.hint or ""
        if hint_text and not hint_text.lower().startswith(("example:", "try:")):
            provisional_step = provisional_step.model_copy(update={"hint": f"Example: {hint_text.lstrip()}"})
        ensured_substeps = _ensure_substeps(provisional_step, index, previous_title)
        normalized_step = provisional_step.model_copy(update={"substeps": ensured_substeps})
        normalized_steps.append(normalized_step)

    if len(normalized_steps) < 2:
        fallback = build_fallback_plan(assignment_request)
        for fallback_step in fallback.steps:
            if len(normalized_steps) >= 2:
                break
            normalized_steps.append(fallback_step)

    if normalized_steps:
        normalized_steps[-1] = _ensure_final_answer_details(normalized_steps[-1], format_values)

    overview = (plan.assignment_overview or assignment_request).strip()
    return TutorGamePlan(assignment_overview=overview, steps=normalized_steps)


def build_fallback_plan(assignment_request: str) -> TutorGamePlan:
    format_values = _build_format_values(assignment_request)
    detected_operations = _detect_operations(assignment_request)
    if detected_operations:
        format_values["primary_operation"] = detected_operations[0]
        if len(detected_operations) > 1:
            format_values["secondary_operation"] = detected_operations[1]
    if {"division", "subtraction"}.issubset(set(detected_operations)):
        return _build_two_step_removal_plan(format_values)
    target_steps = _estimate_plan_length(assignment_request, detected_operations)
    operations = detected_operations[:5] or ["strategy"]
    if len(operations) > target_steps:
        operations = operations[:target_steps]

    steps: list[TutorGameStep] = []
    occurrence_tracker: dict[str, int] = {}
    index = 0
    previous_title: str | None = None

    if len(operations) == 1 and target_steps >= 3 and operations[0] not in {"strategy", "practice"}:
        index += 1
        strategy_step = _build_step_from_spec(
            index,
            "strategy",
            format_values,
            previous_title=None,
            occurrence_index=1,
            operation_name=operations[0],
        )
        steps.append(strategy_step)
        previous_title = strategy_step.title

    for operation in operations:
        occurrence_tracker[operation] = occurrence_tracker.get(operation, 0) + 1
        spec_key = operation if operation in _STEP_TEMPLATES else "practice"
        index += 1
        step = _build_step_from_spec(
            index,
            spec_key,
            format_values,
            previous_title=previous_title,
            occurrence_index=occurrence_tracker[operation],
            operation_name=operation,
        )
        steps.append(step)
        previous_title = step.title
        if len(steps) >= target_steps:
            break

    while len(steps) < target_steps:
        remaining_slots = target_steps - len(steps)
        if remaining_slots == 1:
            filler_kind = "result"
        elif len(steps) == 0:
            filler_kind = "strategy"
        else:
            filler_kind = "check"
        index += 1
        step = _build_step_from_spec(
            index,
            filler_kind,
            format_values,
            previous_title=previous_title,
            occurrence_index=1,
            operation_name=None,
        )
        steps.append(step)
        previous_title = step.title
        if filler_kind == "result":
            break

    trimmed_steps = steps[:target_steps]
    if trimmed_steps:
        trimmed_steps[-1] = _ensure_final_answer_details(trimmed_steps[-1], format_values)

    overview = f"We'll turn {format_values['context']} into a quest with {target_steps} checkpoints."
    return TutorGamePlan(assignment_overview=overview, steps=trimmed_steps)


async def build_tutor_game_plan(dialogue: Dialogue, params: TutorPlanParams) -> TutorGamePlan:
    try:
        plan = await plan_summarizer.run(_PLAN_EXAMPLES, dialogue, params)
        return normalize_tutor_plan(plan, params.assignment_request)
    except Exception:
        return build_fallback_plan(params.assignment_request)
