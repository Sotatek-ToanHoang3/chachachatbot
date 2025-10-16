export interface ChatMessage{
    id: string
    message: string
    is_user: boolean
    metadata?: { [key:string]: any } | undefined
    processing_time?: number | undefined
    timestamp: number
}

export interface TutorGameBridgeStep {
    prompt: string
    hint: string
    success_keywords: string[]
}

export interface TutorGameStep {
    id: string
    title: string
    objective: string
    check_prompt: string
    success_keywords: string[]
    hint: string
    bridge: TutorGameBridgeStep
}

export interface TutorGamePlan {
    assignment_overview: string
    steps: TutorGameStep[]
}
