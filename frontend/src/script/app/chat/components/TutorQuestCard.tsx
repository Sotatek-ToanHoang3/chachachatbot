import { useCallback } from "react"
import { TutorGamePlan } from "src/script/types"
import { storeTutorPlan } from "src/script/app/tutor-game/storage"
import { useDispatch } from "../../../redux/hooks"
import { openTutorGameOverlay } from "../reducer"

const STORAGE_ERROR_MESSAGE = "This browser cannot store the study quest. Try allowing session storage."

export const TutorQuestCard = (props: {
  plan: TutorGamePlan
  sessionId: string
  messageId: string
  question?: string
}) => {
  const dispatch = useDispatch()

  const onPlay = useCallback(() => {
    try {
      storeTutorPlan(props.sessionId, props.messageId, props.plan, props.question)
    } catch (error) {
      console.error(error)
      alert(STORAGE_ERROR_MESSAGE)
      return
    }

    dispatch(
      openTutorGameOverlay({
        sessionId: props.sessionId,
        messageId: props.messageId,
        plan: props.plan,
        question: props.question,
      }),
    )
  }, [dispatch, props.messageId, props.plan, props.question, props.sessionId])

  return (
    <div className="mt-3 rounded-xl bg-white/70 px-4 py-3 shadow-md ring-1 ring-slate-200">
      <p className="text-sm font-semibold text-slate-700">Study Quest Ready!</p>
      {props.question ? (
        <p className="mt-2 rounded-lg bg-indigo-50 px-3 py-2 text-sm font-medium text-indigo-700">
          {props.question}
        </p>
      ) : null}
      <p className="mt-1 text-sm text-slate-600">{props.plan.assignment_overview}</p>
      <ul className="mt-3 space-y-1 text-sm text-slate-700">
        {props.plan.steps.map((step, index) => (
          <li key={step.id} className="flex items-start gap-2">
            <span className="mt-0.5 inline-flex h-5 w-5 items-center justify-center rounded-full bg-indigo-100 text-xs font-semibold text-indigo-600">
              {index + 1}
            </span>
            <div>
              <p className="font-medium text-indigo-700">{step.title}</p>
              <p className="text-slate-600">{step.objective}</p>
              {step.substeps && step.substeps.length > 0 ? (
                <ul className="mt-1 space-y-1 text-xs text-slate-500">
                  {step.substeps.slice(0, 3).map((substep, subIndex) => (
                    <li key={`${step.id}-preview-${subIndex}`}>
                      <span className="font-semibold text-indigo-500">{substep.label}</span>{" "}
                      <span>{substep.prompt}</span>
                    </li>
                  ))}
                </ul>
              ) : null}
            </div>
          </li>
        ))}
      </ul>
      <button
        type="button"
        onClick={onPlay}
        className="mt-4 w-full rounded-lg bg-indigo-500 px-4 py-2 text-sm font-semibold text-white shadow transition hover:bg-indigo-400 focus:outline-none focus:ring-2 focus:ring-indigo-300"
      >
        Play Quest
      </button>
    </div>
  )
}
