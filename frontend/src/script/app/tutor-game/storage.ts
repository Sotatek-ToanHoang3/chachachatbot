import { TutorGamePlan } from "src/script/types"

const STORAGE_PREFIX = "tutor-game-plan"

export function makeTutorPlanKey(sessionId: string, messageId: string): string {
  return `${STORAGE_PREFIX}:${sessionId}:${messageId}`
}

type StoredTutorPlan = {
  plan: TutorGamePlan
  question?: string
}

export function storeTutorPlan(sessionId: string, messageId: string, plan: TutorGamePlan, question?: string): void {
  const key = makeTutorPlanKey(sessionId, messageId)
  const payload: StoredTutorPlan = {
    plan,
    question,
  }
  sessionStorage.setItem(key, JSON.stringify(payload))
}

export function readTutorPlan(sessionId: string, messageId: string): StoredTutorPlan | null {
  const key = makeTutorPlanKey(sessionId, messageId)
  const raw = sessionStorage.getItem(key)
  if (!raw) {
    return null
  }
  try {
    const parsed = JSON.parse(raw)
    if (parsed && typeof parsed === "object" && "plan" in parsed) {
      return parsed as StoredTutorPlan
    }
    return { plan: parsed as TutorGamePlan }
  } catch (error) {
    console.error("Failed to parse stored tutor plan", error)
    sessionStorage.removeItem(key)
    return null
  }
}

export function clearTutorPlan(sessionId: string, messageId: string): void {
  const key = makeTutorPlanKey(sessionId, messageId)
  sessionStorage.removeItem(key)
}
