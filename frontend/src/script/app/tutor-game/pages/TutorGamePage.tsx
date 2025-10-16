import { ChangeEvent, FormEvent, useCallback, useEffect, useMemo, useState } from "react"
import { useParams } from "react-router-dom"
import { nanoid } from "nanoid"

import { NetworkHelper } from "src/script/network"
import { ChatMessage, TutorGamePlan } from "src/script/types"
import { clearTutorPlan, readTutorPlan, storeTutorPlan } from "../storage"

type FeedbackTone = "success" | "hint" | "error" | "bridge"

type AttemptEntry = {
  value: string
  correct: boolean
  bridge: boolean
  matchedKeywords: string[]
}

type LevelState = {
  attempts: number
  succeeded: boolean
  bridgeActivated: boolean
  bridgeSucceeded: boolean
  bridgeAttempts: number
  answers: AttemptEntry[]
  maxMatchedMain: number
  maxMatchedBridge: number
  totalKeywords: number
}

type Feedback = {
  tone: FeedbackTone
  message: string
  matchedKeywords?: string[]
}

type LoadState = "loading" | "ready" | "error"

interface TutorGameLevelReport {
  step_id: string
  title: string
  attempts: number
  bridge_used: boolean
  bridge_attempts: number
  matched_keywords: string[]
  bridge_matched_keywords: string[]
  succeeded: boolean
}

interface TutorGameSummaryReport {
  session_id: string
  source_message_id: string
  assignment_overview: string
  total_levels: number
  completed_levels: number
  score: number
  badges: string[]
  levels: TutorGameLevelReport[]
  started_at: string
  finished_at: string
}

interface SummaryData {
  score: number
  badges: string[]
  report: TutorGameSummaryReport
  userMessage: ChatMessage
  assistantMessage?: ChatMessage
}

export type TutorGamePageProps = {
  sessionId?: string
  messageId?: string
  plan?: TutorGamePlan | null
  embed?: boolean
  question?: string | null
  onRequestClose?: (context: { reason: "quit" | "completed" }) => void
  onSummary?: (data: { userMessage: ChatMessage; assistantMessage?: ChatMessage }) => void
}

const FEEDBACK_STYLE_MAP: Record<FeedbackTone, string> = {
  success: "bg-emerald-100 text-emerald-700 border-emerald-200",
  hint: "bg-amber-100 text-amber-700 border-amber-200",
  error: "bg-rose-100 text-rose-700 border-rose-200",
  bridge: "bg-indigo-100 text-indigo-700 border-indigo-200",
}

function normalizeText(value: string): string {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim()
}

function evaluateAnswer(value: string, keywords: string[]): { success: boolean; matched: string[] } {
  if (keywords.length === 0) {
    return { success: value.trim().length > 0, matched: [] }
  }
  const normalized = normalizeText(value)
  const matched = keywords
    .map(keyword => keyword.toLowerCase().trim())
    .filter(keyword => keyword && normalized.includes(keyword))
  const required = Math.max(1, Math.ceil(keywords.length / 2))
  return {
    success: matched.length >= required,
    matched,
  }
}

function computeScore(states: LevelState[]): number {
  if (states.length === 0) {
    return 0
  }
  const MAX_POINTS_PER_LEVEL = 3
  let totalPoints = 0
  states.forEach(state => {
    const attemptPenalty = Math.max(0, state.attempts - 1)
    const bridgePenalty = state.bridgeActivated ? 1 : 0
    const extraBridgePenalty = Math.max(0, state.bridgeAttempts - 1)
    const levelPoints = Math.max(0, MAX_POINTS_PER_LEVEL - (attemptPenalty + bridgePenalty + extraBridgePenalty))
    totalPoints += levelPoints
  })
  const maxPoints = states.length * MAX_POINTS_PER_LEVEL
  return Math.round((totalPoints / maxPoints) * 100)
}

function computeBadges(plan: TutorGamePlan, states: LevelState[]): string[] {
  if (plan.steps.length === 0) {
    return ["Curious Explorer"]
  }
  const badges = new Set<string>()
  if (states.every(state => state.attempts === 1 && !state.bridgeActivated)) {
    badges.add("Lightning Thinker")
  }
  if (states.some(state => state.bridgeActivated)) {
    badges.add("Bounce-Back Hero")
  }
  if (states.some(state => state.attempts > 1)) {
    badges.add("Persistence Pro")
  }
  if (states.some(state => state.maxMatchedMain >= state.totalKeywords && state.totalKeywords > 0)) {
    badges.add("Keyword Detective")
  }
  if (badges.size === 0) {
    badges.add("Curious Explorer")
  }
  return Array.from(badges)
}

function extractNumbersFromText(text: string): string[] {
  const matches = text.match(/\d+/g) ?? []
  const unique = new Set<string>()
  matches.forEach(match => {
    const normalized = match.replace(/^0+(?=\d)/, "")
    if (normalized.length > 0) {
      unique.add(normalized)
    }
  })
  return Array.from(unique)
}

function fallbackNumberSuggestions(seed: number): string[] {
  const presets = [
    ["4", "8", "12", "16", "20"],
    ["3", "6", "9", "12", "18"],
    ["5", "10", "15", "20", "25"],
    ["7", "14", "21", "28", "35"],
  ]
  return presets[seed % presets.length]
}

function buildSummaryReport(
  plan: TutorGamePlan,
  states: LevelState[],
  sessionId: string,
  messageId: string,
  startedAt: string,
  score: number,
  badges: string[],
): TutorGameSummaryReport {
  const finishedAt = new Date().toISOString()
  const levels: TutorGameLevelReport[] = plan.steps.map((step, index) => {
    const state = states[index]
    const matchedMain = new Set<string>()
    const matchedBridge = new Set<string>()
    state.answers.forEach(entry => {
      if (entry.bridge) {
        entry.matchedKeywords.forEach(keyword => matchedBridge.add(keyword))
      } else {
        entry.matchedKeywords.forEach(keyword => matchedMain.add(keyword))
      }
    })
    return {
      step_id: step.id,
      title: step.title,
      attempts: state.attempts,
      bridge_used: state.bridgeActivated,
      bridge_attempts: state.bridgeAttempts,
      matched_keywords: Array.from(matchedMain),
      bridge_matched_keywords: Array.from(matchedBridge),
      succeeded: state.succeeded,
    }
  })

  return {
    session_id: sessionId,
    source_message_id: messageId,
    assignment_overview: plan.assignment_overview,
    total_levels: plan.steps.length,
    completed_levels: states.filter(state => state.succeeded).length,
    score,
    badges,
    levels,
    started_at: startedAt,
    finished_at: finishedAt,
  }
}

function createInitialStates(plan: TutorGamePlan): LevelState[] {
  return plan.steps.map(step => ({
    attempts: 0,
    succeeded: false,
    bridgeActivated: false,
    bridgeSucceeded: false,
    bridgeAttempts: 0,
    answers: [],
    maxMatchedMain: 0,
    maxMatchedBridge: 0,
    totalKeywords: step.success_keywords?.length ?? 0,
  }))
}

export const TutorGamePage = (props: TutorGamePageProps) => {
  const params = useParams()
  const sessionId = props.sessionId ?? params.sessionId ?? ""
  const messageId = props.messageId ?? params.messageId ?? ""
  const isEmbedded = props.embed === true
  const initialPlan = props.plan ?? null

  const [loadState, setLoadState] = useState<LoadState>(initialPlan ? "ready" : "loading")
  const [plan, setPlan] = useState<TutorGamePlan | null>(initialPlan)
  const [question, setQuestion] = useState<string | undefined>(props.question ?? undefined)
  const [levelStates, setLevelStates] = useState<LevelState[]>(() => (initialPlan ? createInitialStates(initialPlan) : []))
  const [currentIndex, setCurrentIndex] = useState(0)
  const [inputValue, setInputValue] = useState("")
  const [bridgeActive, setBridgeActive] = useState(false)
  const [awaitingAdvance, setAwaitingAdvance] = useState(false)
  const [feedback, setFeedback] = useState<Feedback | null>(null)
  const [gameFinished, setGameFinished] = useState(false)
  const [summaryData, setSummaryData] = useState<SummaryData | null>(null)
  const [summaryError, setSummaryError] = useState<string | null>(null)
  const [isSubmittingSummary, setIsSubmittingSummary] = useState(false)
  const [finalStates, setFinalStates] = useState<LevelState[] | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [startedAt] = useState(() => new Date().toISOString())

  const totalLevels = plan?.steps.length ?? 0

  useEffect(() => {
    if (props.question) {
      setQuestion(props.question || undefined)
    }
  }, [props.question])

  useEffect(() => {
    let didCancel = false
    const resolvePlan = async () => {
      if (!sessionId || !messageId) {
        setLoadError("Missing session information.")
        setLoadState("error")
        return
      }

      if (props.plan) {
        if (!didCancel) {
          setPlan(props.plan)
          setLevelStates(createInitialStates(props.plan))
          setQuestion(props.question ?? undefined)
          setLoadState("ready")
          try {
            storeTutorPlan(sessionId, messageId, props.plan, props.question ?? undefined)
          } catch (error) {
            console.warn("Unable to cache tutor plan in session storage.", error)
          }
        }
        return
      }

      try {
        const fromStorage = readTutorPlan(sessionId, messageId)
        if (fromStorage) {
          if (!didCancel) {
            setPlan(fromStorage.plan)
            setLevelStates(createInitialStates(fromStorage.plan))
            setQuestion(fromStorage.question ?? props.question ?? undefined)
            setLoadState("ready")
          }
          return
        }
        const messages = await NetworkHelper.loadSessionChatMessages(sessionId)
        const target = messages.find(turn => turn.id === messageId)
        const planFromChat = target?.metadata?.tutor_game_plan as TutorGamePlan | undefined
        if (planFromChat) {
          const questionFromChat: string | undefined =
            target?.metadata?.tutor_assignment_prompt ??
            target?.metadata?.assignment_request ??
            target?.metadata?.assignment_initial_prompt ??
            undefined
          storeTutorPlan(sessionId, messageId, planFromChat, questionFromChat)
          if (!didCancel) {
            setPlan(planFromChat)
            setLevelStates(createInitialStates(planFromChat))
            setQuestion(questionFromChat ?? props.question ?? undefined)
            setLoadState("ready")
          }
        } else {
          throw new Error("Quest plan not found in chat history.")
        }
      } catch (error) {
        console.error(error)
        if (!didCancel) {
          setLoadError("We couldn't recover the study quest. Please reopen it from the chat.")
          setLoadState("error")
        }
      }
    }

    resolvePlan()
    return () => {
      didCancel = true
    }
  }, [messageId, props.plan, sessionId])

  const completedLevels = useMemo(
    () => levelStates.filter(state => state.succeeded).length,
    [levelStates],
  )

  const progressPercent = useMemo(() => {
    if (!plan || plan.steps.length === 0) {
      return 0
    }
    return Math.round((completedLevels / plan.steps.length) * 100)
  }, [plan, completedLevels])

  const currentStep = plan ? plan.steps[currentIndex] : null
  const currentState = levelStates[currentIndex]
  const isLastLevel = plan ? currentIndex === plan.steps.length - 1 : false

  const numberSuggestions = useMemo(() => {
    if (!plan || !currentStep) {
      return []
    }
    const stepText = [
      currentStep.check_prompt,
      currentStep.objective,
      currentStep.hint,
      currentStep.bridge?.prompt ?? "",
      currentStep.bridge?.hint ?? "",
    ].join(" ")
    const planContext = plan.steps.map(step => `${step.check_prompt} ${step.objective}`).join(" ")
    const collected = [
      ...extractNumbersFromText(stepText),
      ...extractNumbersFromText(plan.assignment_overview),
      ...extractNumbersFromText(planContext),
    ]
    const unique = Array.from(new Set(collected))
    if (unique.length >= 3) {
      return unique.slice(0, 6)
    }
    const fallback = fallbackNumberSuggestions(currentIndex)
    const merged = Array.from(new Set([...unique, ...fallback]))
    return merged.slice(0, 6)
  }, [currentIndex, currentStep, plan])

  const starterSuggestions = useMemo(() => {
    if (!currentStep) {
      return []
    }
    const starters: string[] = []
    if (question) {
      starters.push(`The question asks: ${question}`)
    }
    if (currentStep.objective) {
      const objective = currentStep.objective.replace(/\.$/, "")
      starters.push(`I need to ${objective.toLowerCase()}.`)
    }
    if (currentStep.check_prompt) {
      starters.push(currentStep.check_prompt)
    }
    if (currentStep.bridge?.prompt) {
      starters.push(`Bridge idea: ${currentStep.bridge.prompt}`)
    }
    return Array.from(new Set(starters.filter(Boolean))).slice(0, 5)
  }, [currentStep, question])

  const handleInsertSnippet = useCallback((snippet: string) => {
    setInputValue(prev => {
      const trimmed = prev.trimEnd()
      if (!trimmed) {
        return snippet
      }
      const needsSpace = /\s$/.test(prev)
      return `${needsSpace ? prev : `${trimmed} `}${snippet}`
    })
  }, [])

  const activityIdeas = useMemo(() => {
    if (!currentStep) {
      return []
    }
    const ideas: Array<{ title: string; description: string }> = []

    if (currentIndex === 0) {
      ideas.push({
        title: "Highlight the clues",
        description: "List or underline the important numbers and labels you see.",
      })
    } else if (plan && currentIndex === plan.steps.length - 1) {
      ideas.push({
        title: "Check your answer",
        description: "Explain why your result makes sense or try a quick reverse check.",
      })
    } else {
      ideas.push({
        title: "Map the steps",
        description: "Say which math move you plan to use and why it fits.",
      })
    }

    if (currentStep.bridge?.prompt) {
      ideas.push({
        title: "Bridge boost",
        description: currentStep.bridge.prompt,
      })
    }

    if (currentStep.success_keywords?.length) {
      const sample = currentStep.success_keywords.slice(0, 3).join(", ")
      ideas.push({
        title: "Use key words",
        description: `Try including words like ${sample}.`,
      })
    }

    if (question) {
      ideas.push({
        title: "Connect to the story",
        description: "Mention what the question is asking so your answer stays on track.",
      })
    }

    return ideas.slice(0, 3)
  }, [currentIndex, currentStep, plan, question])

  const handleInputChange = useCallback((event: ChangeEvent<HTMLTextAreaElement>) => {
    setInputValue(event.target.value)
  }, [])

  const finishGame = useCallback(
    async (statesSnapshot: LevelState[]) => {
      if (!plan || !sessionId || !messageId || isSubmittingSummary) {
        return
      }
      setSummaryError(null)
      setIsSubmittingSummary(true)
      const score = computeScore(statesSnapshot)
      const badges = computeBadges(plan, statesSnapshot)
      const report = buildSummaryReport(plan, statesSnapshot, sessionId, messageId, startedAt, score, badges)
      const userMessage: ChatMessage = {
        id: nanoid(),
        message: "I finished the study quest! Here’s my progress report.",
        is_user: true,
        metadata: {
          tutor_game_summary: report,
        },
        timestamp: Date.now(),
      }

      try {
        const assistantMessage = await NetworkHelper.sendUserMessage(sessionId, userMessage)
        setSummaryData({
          score,
          badges,
          report,
          userMessage,
          assistantMessage,
        })
        props.onSummary?.({ userMessage, assistantMessage })
        clearTutorPlan(sessionId, messageId)
        setIsSubmittingSummary(false)
        setGameFinished(true)
        const origin = window.location.origin
        if (window.opener) {
          try {
            window.opener.postMessage(
              {
                type: "TUTOR_GAME_SUMMARY",
                payload: {
                  userMessage,
                  assistantMessage,
                },
              },
              origin,
            )
          } catch (error) {
            console.warn("Unable to notify parent window about tutor game summary.", error)
          }
        }
      } catch (error) {
        console.error("Failed to send tutor game summary", error)
        setSummaryError("We couldn't send the summary back to chat. Check your connection and try again.")
        setSummaryData({
          score,
          badges,
          report,
          userMessage,
        })
        setIsSubmittingSummary(false)
        setGameFinished(true)
      }
    },
    [isSubmittingSummary, messageId, plan, props.onSummary, sessionId, startedAt],
  )

  const handleSubmit = useCallback(
    (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault()
      if (!plan || !currentStep || !currentState || awaitingAdvance || gameFinished || isSubmittingSummary) {
        return
      }
      const text = inputValue.trim()
      if (!text) {
        setFeedback({
          tone: "error",
          message: "Type your idea before checking.",
        })
        return
      }

      const keywords = bridgeActive ? currentStep.bridge.success_keywords : currentStep.success_keywords
      const evaluation = evaluateAnswer(text, keywords)
      const nextAttempts = currentState.attempts + (bridgeActive ? 0 : 1)
      const shouldActivateBridge =
        !evaluation.success && !bridgeActive && nextAttempts >= 2 && !currentState.bridgeActivated

      const updatedStates = levelStates.map((state, index) => {
        if (index !== currentIndex) {
          return state
        }
        if (bridgeActive) {
          return {
            ...state,
            bridgeAttempts: state.bridgeAttempts + 1,
            bridgeActivated: true,
            bridgeSucceeded: state.bridgeSucceeded || evaluation.success,
            answers: [
              ...state.answers,
              {
                value: text,
                correct: evaluation.success,
                bridge: true,
                matchedKeywords: evaluation.matched,
              },
            ],
            maxMatchedBridge: Math.max(state.maxMatchedBridge, evaluation.matched.length),
          }
        }
        return {
          ...state,
          attempts: state.attempts + 1,
          succeeded: evaluation.success ? true : state.succeeded,
          bridgeActivated: state.bridgeActivated || shouldActivateBridge,
          answers: [
            ...state.answers,
            {
              value: text,
              correct: evaluation.success,
              bridge: false,
              matchedKeywords: evaluation.matched,
            },
          ],
          maxMatchedMain: Math.max(state.maxMatchedMain, evaluation.matched.length),
        }
      })

      setLevelStates(updatedStates)

      if (bridgeActive) {
        if (evaluation.success) {
          setBridgeActive(false)
          setInputValue("")
          setFeedback({
            tone: "success",
            message: "Bridge cleared! Use that idea to answer the main challenge.",
          })
        } else {
          setFeedback({
            tone: "hint",
            message: currentStep.bridge.hint,
          })
        }
        return
      }

      if (evaluation.success) {
        const successMessage =
          evaluation.matched.length > 0
            ? `Nice! You used: ${evaluation.matched.join(", ")}`
            : "Great work! That fits the step."
        setFeedback({
          tone: "success",
          message: successMessage,
          matchedKeywords: evaluation.matched,
        })
        setInputValue("")
        setBridgeActive(false)
        if (isLastLevel) {
          setFinalStates(updatedStates)
          finishGame(updatedStates)
        } else {
          setAwaitingAdvance(true)
        }
        return
      }

      if (shouldActivateBridge) {
        setBridgeActive(true)
        setFeedback({
          tone: "bridge",
          message: `Bridge step unlocked: ${currentStep.bridge.prompt}`,
        })
        setInputValue("")
      } else if (currentState.bridgeActivated) {
        setFeedback({
          tone: "hint",
          message: currentStep.bridge.hint,
        })
        setInputValue("")
      } else {
        setFeedback({
          tone: "hint",
          message: currentStep.hint,
        })
        setInputValue("")
      }
    },
    [
      awaitingAdvance,
      bridgeActive,
      currentIndex,
      currentState,
      currentStep,
      finishGame,
      gameFinished,
      inputValue,
      isLastLevel,
      isSubmittingSummary,
      levelStates,
      plan,
    ],
  )

  const handleAdvance = useCallback(() => {
    if (!plan) {
      return
    }
    const nextIndex = Math.min(currentIndex + 1, plan.steps.length - 1)
    setCurrentIndex(nextIndex)
    setAwaitingAdvance(false)
    setBridgeActive(false)
    setInputValue("")
    setFeedback(null)
  }, [currentIndex, plan])

  const handleRetrySend = useCallback(() => {
    if (finalStates) {
      finishGame(finalStates)
    }
  }, [finalStates, finishGame])

  const handleCloseWindow = useCallback(() => {
    if (isEmbedded) {
      props.onRequestClose?.({ reason: gameFinished ? "completed" : "quit" })
      return
    }
    if (window.opener) {
      try {
        window.opener.focus()
      } catch {
        // ignore
      }
    }
    window.close()
  }, [gameFinished, isEmbedded, props.onRequestClose])

  if (loadState === "loading") {
    if (isEmbedded) {
      return (
        <div className="flex h-full items-center justify-center bg-white">
          <div className="rounded-2xl bg-white px-6 py-8 text-center shadow-lg ring-1 ring-indigo-100">
            <p className="text-lg font-semibold text-indigo-600">Loading your study quest...</p>
            <p className="mt-2 text-sm text-slate-500">Hang tight while we get everything ready.</p>
          </div>
        </div>
      )
    }
    return (
      <div className="flex min-h-screen items-center justify-center bg-indigo-50">
        <div className="rounded-2xl bg-white px-6 py-8 text-center shadow-lg">
          <p className="text-lg font-semibold text-indigo-600">Loading your study quest...</p>
          <p className="mt-2 text-sm text-slate-500">Hang tight while we get everything ready.</p>
        </div>
      </div>
    )
  }

  if (loadState === "error" || !plan || !currentStep) {
    if (isEmbedded) {
      return (
        <div className="flex h-full items-center justify-center bg-white">
          <div className="rounded-2xl bg-white px-6 py-8 text-center shadow-lg ring-1 ring-rose-100">
            <p className="text-lg font-semibold text-rose-600">Quest unavailable</p>
            <p className="mt-2 text-sm text-slate-600">{loadError ?? "Please return to the chat and try again."}</p>
            <button
              type="button"
              onClick={handleCloseWindow}
              className="mt-4 rounded-lg bg-rose-500 px-4 py-2 text-sm font-semibold text-white shadow hover:bg-rose-400"
            >
              Return to chat
            </button>
          </div>
        </div>
      )
    }
    return (
      <div className="flex min-h-screen items-center justify-center bg-rose-50">
        <div className="rounded-2xl bg-white px-6 py-8 text-center shadow-lg">
          <p className="text-lg font-semibold text-rose-600">Quest unavailable</p>
          <p className="mt-2 text-sm text-slate-600">{loadError ?? "Please return to the chat and try again."}</p>
          <button
            type="button"
            onClick={handleCloseWindow}
            className="mt-4 rounded-lg bg-rose-500 px-4 py-2 text-sm font-semibold text-white shadow hover:bg-rose-400"
          >
            Close
          </button>
        </div>
      </div>
    )
  }

  if (gameFinished && summaryData) {
    return (
      <div
        className={
          isEmbedded
            ? "flex h-full flex-col overflow-y-auto bg-gradient-to-b from-indigo-50 via-white to-blue-100"
            : "min-h-screen bg-gradient-to-b from-indigo-100 via-white to-blue-100"
        }
      >
        <div className={`mx-auto max-w-3xl px-4 ${isEmbedded ? "py-6" : "py-10"}`}>
          <div className="rounded-3xl bg-white px-6 py-8 shadow-xl ring-1 ring-indigo-100">
            <p className="text-sm font-semibold uppercase tracking-wide text-indigo-500">Quest Complete</p>
            <h1 className="mt-2 text-3xl font-bold text-slate-900">You finished the study quest!</h1>
            <p className="mt-2 text-sm text-slate-600">
              Your summary was {summaryError ? "not sent yet" : "shared"} with CHACHA. Check the chat to continue
              planning next steps together.
            </p>
            {question ? (
              <div className="mt-4 rounded-2xl border border-indigo-100 bg-indigo-50/80 px-4 py-3">
                <p className="text-xs font-semibold uppercase tracking-wide text-indigo-500">Question solved</p>
                <p className="mt-1 text-sm text-slate-800">{question}</p>
              </div>
            ) : null}
            <div className="mt-6 flex flex-col gap-6 rounded-2xl bg-slate-50 p-6 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <p className="text-sm font-semibold uppercase tracking-wide text-slate-500">Score</p>
                <p className="text-5xl font-bold text-indigo-600">{summaryData.score}</p>
              </div>
              <div className="flex flex-wrap gap-2">
                {summaryData.badges.map(badge => (
                  <span
                    key={badge}
                    className="inline-flex items-center rounded-full bg-indigo-100 px-3 py-1 text-sm font-medium text-indigo-600"
                  >
                    {badge}
                  </span>
                ))}
              </div>
            </div>
            <div className="mt-6">
              <h2 className="text-lg font-semibold text-slate-900">Level breakdown</h2>
              <ul className="mt-3 space-y-3">
                {summaryData.report.levels.map(level => (
                  <li key={level.step_id} className="rounded-xl border border-slate-200 bg-white p-4">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div>
                        <p className="text-sm font-semibold text-indigo-600">{level.title}</p>
                        <p className="text-xs text-slate-500">
                          Attempts: {level.attempts} {level.bridge_used ? "· Bridge used" : ""}
                        </p>
                      </div>
                      <span
                        className={`inline-flex items-center rounded-full px-3 py-1 text-xs font-semibold ${
                          level.succeeded ? "bg-emerald-100 text-emerald-600" : "bg-rose-100 text-rose-600"
                        }`}
                      >
                        {level.succeeded ? "Cleared" : "In progress"}
                      </span>
                    </div>
                    {level.matched_keywords.length > 0 ? (
                      <p className="mt-2 text-xs text-slate-500">
                        Keywords hit: {level.matched_keywords.join(", ")}
                      </p>
                    ) : null}
                  </li>
                ))}
              </ul>
            </div>
            {summaryError ? (
              <p className="mt-4 rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-600">
                {summaryError}
              </p>
            ) : null}
            <div className="mt-8 flex flex-wrap gap-3">
              {summaryError ? (
                <button
                  type="button"
                  onClick={handleRetrySend}
                  className="rounded-lg bg-indigo-500 px-4 py-2 text-sm font-semibold text-white shadow hover:bg-indigo-400"
                >
                  Retry sending summary
                </button>
              ) : null}
              <button
                type="button"
                onClick={handleCloseWindow}
                className="rounded-lg bg-slate-800 px-4 py-2 text-sm font-semibold text-white shadow hover:bg-slate-700"
              >
                {isEmbedded ? "Return to chat" : "Close window"}
              </button>
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div
      className={
        isEmbedded
          ? "flex h-full flex-col overflow-y-auto bg-gradient-to-b from-indigo-50 via-white to-blue-100"
          : "min-h-screen bg-gradient-to-b from-indigo-100 via-white to-blue-100"
      }
    >
      <div className={`mx-auto max-w-3xl px-4 ${isEmbedded ? "py-6" : "py-10"}`}>
        <div className="rounded-3xl bg-white px-6 py-8 shadow-xl ring-1 ring-indigo-100">
          <p className="text-sm font-semibold uppercase tracking-wide text-indigo-500">Study Quest</p>
          <h1 className="mt-2 text-3xl font-bold text-slate-900">{plan.assignment_overview}</h1>
          {question ? (
            <div className="mt-4 rounded-2xl border border-indigo-100 bg-indigo-50/80 px-4 py-3">
              <p className="text-xs font-semibold uppercase tracking-wide text-indigo-500">Original Question</p>
              <p className="mt-1 text-sm text-slate-800">{question}</p>
            </div>
          ) : null}
          <div className="mt-4">
            <div className="h-2 w-full rounded-full bg-slate-200">
              <div
                className="h-full rounded-full bg-indigo-500 transition-all"
                style={{ width: `${progressPercent}%` }}
              />
            </div>
            <p className="mt-2 text-xs font-medium uppercase tracking-wide text-slate-500">
              Level {currentIndex + 1} of {plan.steps.length} · {progressPercent}% complete
            </p>
          </div>
          <div className="mt-6 rounded-2xl border border-slate-200 bg-slate-50 p-6">
            <p className="text-sm font-semibold uppercase tracking-wide text-indigo-600">Current Level</p>
            <h2 className="mt-1 text-2xl font-semibold text-slate-900">
              Level {currentIndex + 1}: {currentStep.title}
            </h2>
            <p className="mt-2 text-sm text-slate-600">{currentStep.objective}</p>
            <div className="mt-4 rounded-xl border border-indigo-100 bg-white px-4 py-3">
              <p className="text-sm font-semibold text-indigo-600">
                {bridgeActive ? "Bridge Prompt" : "Challenge Prompt"}
              </p>
              <p className="mt-1 text-sm text-slate-700">
                {bridgeActive ? currentStep.bridge.prompt : currentStep.check_prompt}
              </p>
            </div>
            {feedback ? (
              <div className={`mt-4 rounded-xl border px-4 py-3 text-sm ${FEEDBACK_STYLE_MAP[feedback.tone]}`}>
                <p>{feedback.message}</p>
              </div>
            ) : null}
            <form className="mt-5" onSubmit={handleSubmit}>
              <label htmlFor="tutor-game-answer" className="sr-only">
                Your answer
              </label>
              <textarea
                id="tutor-game-answer"
                value={inputValue}
                onChange={handleInputChange}
                rows={3}
                className="w-full rounded-xl border border-slate-200 px-3 py-2 text-sm shadow-sm focus:border-indigo-400 focus:outline-none focus:ring-2 focus:ring-indigo-200"
                placeholder="Type your answer here..."
                disabled={awaitingAdvance || isSubmittingSummary}
              />
              <div className="mt-4 flex flex-wrap items-center gap-3">
                <button
                  type="submit"
                  disabled={awaitingAdvance || isSubmittingSummary}
                  className="rounded-lg bg-indigo-500 px-4 py-2 text-sm font-semibold text-white shadow hover:bg-indigo-400 disabled:cursor-not-allowed disabled:bg-indigo-300"
                >
                  Check answer
                </button>
                {awaitingAdvance ? (
                  <button
                    type="button"
                    onClick={handleAdvance}
                    className="rounded-lg bg-emerald-500 px-4 py-2 text-sm font-semibold text-white shadow hover:bg-emerald-400"
                  >
                    Next level
                  </button>
                ) : null}
                {isSubmittingSummary ? (
                  <span className="text-xs font-medium uppercase tracking-wide text-slate-500">
                    Sending summary to chat...
                  </span>
                ) : null}
              </div>
            </form>
            {starterSuggestions.length > 0 ? (
              <div className="mt-4 rounded-xl border border-indigo-100 bg-white px-4 py-3">
                <p className="text-xs font-semibold uppercase tracking-wide text-indigo-500">Quick Starters</p>
                <p className="mt-1 text-xs text-slate-600">Tap a prompt to drop helpful wording into your answer.</p>
                <div className="mt-3 flex flex-wrap gap-2">
                  {starterSuggestions.map(starter => (
                    <button
                      type="button"
                      key={starter}
                      onClick={() => handleInsertSnippet(starter)}
                      className="rounded-full border border-indigo-200 px-3 py-1 text-xs font-semibold text-slate-700 shadow-sm transition hover:bg-indigo-50 focus:outline-none focus:ring-2 focus:ring-indigo-300"
                    >
                      {starter}
                    </button>
                  ))}
                </div>
              </div>
            ) : null}
            {numberSuggestions.length > 0 ? (
              <div className="mt-4 rounded-xl border border-indigo-100 bg-indigo-50/40 px-4 py-3">
                <p className="text-xs font-semibold uppercase tracking-wide text-indigo-500">Number ideas</p>
                <p className="mt-1 text-xs text-slate-600">
                  Tap a number to drop it into your answer, then explain how it fits.
                </p>
                <div className="mt-3 flex flex-wrap gap-2">
                  {numberSuggestions.map(suggestion => (
                    <button
                      type="button"
                      key={suggestion}
                      onClick={() => handleInsertSnippet(suggestion)}
                      className="rounded-full border border-indigo-200 px-3 py-1 text-xs font-semibold text-indigo-600 shadow-sm transition hover:bg-indigo-50 focus:outline-none focus:ring-2 focus:ring-indigo-300"
                    >
                      {suggestion}
                    </button>
                  ))}
                </div>
              </div>
            ) : null}
            {activityIdeas.length > 0 ? (
              <div className="mt-5 grid gap-3 sm:grid-cols-2">
                {activityIdeas.map(idea => (
                  <div key={idea.title} className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 shadow-sm">
                    <p className="text-sm font-semibold text-indigo-600">{idea.title}</p>
                    <p className="mt-1 text-xs text-slate-600">{idea.description}</p>
                  </div>
                ))}
              </div>
            ) : null}
            <p className="mt-5 text-xs text-slate-500">
              Tip: Using keywords such as <em>{currentStep.success_keywords.join(", ")}</em> will help your answer pass.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default TutorGamePage
