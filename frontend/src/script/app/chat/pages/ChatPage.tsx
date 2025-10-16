import { IntroView } from "../components/IntroView";
import { useEffect, useRef, useState } from "react";
import { nanoid } from "nanoid";
import { useDispatch, useSelector } from "../../../redux/hooks";
import { BackgroundPanel } from "src/script/components/background";
import { useParams } from "react-router-dom";

import { yupResolver } from "@hookform/resolvers/yup"
import { EntityId } from "@reduxjs/toolkit"
import { useCallback, useMemo, KeyboardEvent, FocusEvent } from "react"
import { useForm } from "react-hook-form"
import * as yup from "yup"
import { addMessage, closeTutorGameOverlay, loadChatSession, regenerateLastSystemMessage, sendUserMessage } from "../reducer"
import { MessageView } from "src/script/components/messages"
import { CopyToClipboard } from 'react-copy-to-clipboard';
import path from "path"
import { ClipboardDocumentIcon, PaperAirplaneIcon } from "@heroicons/react/20/solid";
import { enqueueSnackbar } from "notistack"
import TextareaAutosize from 'react-textarea-autosize';
import { useMediaQuery } from "react-responsive"
import { useOnScreenKeyboardScrollFix, useViewportSize } from "src/script/mobile-utils"
import { SessionInfoPanel } from "../../../components/SessionInfoPanel"
import { EmotionPicker } from "../components/EmotionPicker";
import { TutorBadge } from "../components/TutorBadge";
import { TutorQuestCard } from "../components/TutorQuestCard";
import { TutorGamePage } from "src/script/app/tutor-game/pages/TutorGamePage";
import useAsyncEffect from 'use-async-effect';
import { NetworkHelper } from "src/script/network";
import { useTranslation } from "react-i18next";
import { ChatMessage, TutorGamePlan } from "src/script/types";

export const ChatPage = () => {

  const { sessionId } = useParams()

  const sessionInfoExists = useSelector(state => state.chatState.sessionInfo != null)

  const dispatch = useDispatch()

  useAsyncEffect(async isMounted => {
    if(sessionId != null){
      try{
        const sessionInfo = await NetworkHelper.loadSessionInfo(sessionId)
        if(isMounted() == true) {
        // session info exists.
        dispatch(loadChatSession(sessionId, true))
      }
    }catch(ex){

    }
  }
}, [sessionId])


  return <>
    {
      sessionInfoExists ? <ChatView /> : <IntroView sessionId={sessionId!}/>
    }
    <BackgroundPanel />
  </>
}


const mobileMediaQuery = { minWidth: 640 }
function useIsMobile(): boolean{
  return useMediaQuery(mobileMediaQuery) === false
}

const ChatView = () => {

  const desktopScrollViewRef = useRef<HTMLDivElement>(null)
  const mobileScrollViewRef = useRef<HTMLDivElement>(null)

  const isMobile = useIsMobile()
  const dispatch = useDispatch()

  useOnScreenKeyboardScrollFix(isMobile)


  const messageIds = useSelector(state => state.chatState.messages.ids)
  const messagesById = useSelector(state => state.chatState.messages.entities)
  const activeTutorGame = useSelector(state => state.chatState.activeTutorGame)

  const [_, viewPortHeight] = useViewportSize()

  const handleTutorGameSummary = useCallback(
    (data: { userMessage: ChatMessage; assistantMessage?: ChatMessage }) => {
      if (data.userMessage) {
        dispatch(addMessage(data.userMessage))
      }
      if (data.assistantMessage) {
        dispatch(addMessage(data.assistantMessage))
      }
    },
    [dispatch],
  )

  const handleTutorGameClose = useCallback(
    (context: { reason: "quit" | "completed" }) => {
      const currentGame = activeTutorGame
      dispatch(closeTutorGameOverlay())
      if (context.reason === "quit" && currentGame) {
        dispatch(
          sendUserMessage({
            id: nanoid(),
            message: "I'm going to pause the study quest for now.",
            is_user: true,
            metadata: {
              tutor_game_status: {
                status: "paused",
                source_message_id: currentGame.messageId,
              },
            },
            timestamp: Date.now(),
          }),
        )
      }
    },
    [activeTutorGame, dispatch],
  )

  const scrollToBottom = useCallback(() => {

    const scrollViewRef = isMobile === true ? mobileScrollViewRef : desktopScrollViewRef
    if (scrollViewRef?.current != null) {
      const scroll = scrollViewRef.current.scrollHeight -
        scrollViewRef.current.clientHeight;
      scrollViewRef.current.scrollTo({
        behavior: "smooth",
        top: scroll
      })
    }
  }, [isMobile])

  const onTypingPanelFocus = useCallback(()=>{
    
    requestAnimationFrame(()=>{
      if(isMobile === true){
        setTimeout(scrollToBottom, 200)
      }
    })
  }, [scrollToBottom, isMobile])

  useEffect(() => {
    requestAnimationFrame(() => {
      scrollToBottom()
    })
  }, [messageIds.length])

  useEffect(() => {
    const origin = window.location.origin
    const handleMessage = (event: MessageEvent) => {
      if (event.origin !== origin) {
        return
      }
      const data = event.data as { type?: string; payload?: { userMessage?: any; assistantMessage?: any } } | undefined
      if (!data || data.type !== "TUTOR_GAME_SUMMARY" || !data.payload) {
        return
      }
      dispatch((innerDispatch, getState) => {
        const existingIds = new Set<string>(getState().chatState.messages.ids.map((id: any) => id.toString()))
        const maybeDispatch = (message: any) => {
          if (!message || typeof message !== "object") {
            return
          }
          if (message.id && !existingIds.has(message.id.toString())) {
            innerDispatch(addMessage(message))
          }
        }
        const payload = data.payload!
        maybeDispatch(payload.userMessage)
        maybeDispatch(payload.assistantMessage)
        const summarySource =
          payload.userMessage?.metadata?.tutor_game_summary?.source_message_id ??
          payload.assistantMessage?.metadata?.tutor_game_summary?.source_message_id
        if (summarySource) {
          const activeGame = getState().chatState.activeTutorGame
          if (activeGame && activeGame.messageId === summarySource) {
            innerDispatch(closeTutorGameOverlay())
          }
        }
      })
    }
    window.addEventListener("message", handleMessage)
    return () => window.removeEventListener("message", handleMessage)
  }, [dispatch])

  useEffect(() => {
    if (!activeTutorGame) {
      return
    }
    if (messageIds.length === 0) {
      return
    }
    const latestId = messageIds[messageIds.length - 1]
    const latestMessage = messagesById[latestId]
    if (!latestMessage) {
      return
    }
    const tutorStage = latestMessage.metadata?.tutor_stage
    if (tutorStage && tutorStage !== "quest_ready") {
      dispatch(closeTutorGameOverlay())
      return
    }
    if (
      latestMessage.metadata?.tutor_game_plan &&
      latestMessage.id !== activeTutorGame.messageId
    ) {
      dispatch(closeTutorGameOverlay())
    }
  }, [activeTutorGame, dispatch, messageIds, messagesById])

  return <>
    <div
      style={isMobile === true ? { maxHeight: viewPortHeight, height: viewPortHeight, minHeight: viewPortHeight } : undefined}
      className="overflow-hidden turn-list-container sm:overflow-y-auto justify-end h-screen sm:h-full flex flex-col sm:block"
      ref={desktopScrollViewRef}
    >
      <ChatSessionInfoPanel />
      <div
        className="turn-list container mx-auto px-3 sm:px-10 flex-1 overflow-y-auto sm:overflow-visible"
        ref={mobileScrollViewRef}
      >
        {
          messageIds.map((id, i) => {
            return <SessionMessageView key={id.toString()} id={id} isLast={messageIds.length - 1 === i} />
          })
        }
      </div>
      <TypingPanel onFocus={onTypingPanelFocus} />
    </div>
    {
      activeTutorGame && activeTutorGame.plan
        ? <TutorGameOverlay
            sessionId={activeTutorGame.sessionId}
            messageId={activeTutorGame.messageId}
            plan={activeTutorGame.plan}
            question={activeTutorGame.question}
            onRequestClose={handleTutorGameClose}
            onSummary={handleTutorGameSummary}
          />
        : null
    }
  </>
}

const ChatSessionInfoPanel = () => {
  const sessionInfo = useSelector(state => state.chatState.sessionInfo)

  return <SessionInfoPanel sessionId={sessionInfo!.sessionId} name={sessionInfo!.name} age={sessionInfo!.age}>
    <ShareButton/>
  </SessionInfoPanel>
}


const schema = yup.object({
  message: yup.string().trim().transform((text:string) => text.replace(/ +/g, " ").replace(/[\r\n]+/g, "\n")).required()
}).required()

const TypingPanel = (props: {
  onFocus?: ()=>void,
  onBlur?: ()=>void
}) => {

  const isSystemMessageLoading = useSelector(state => state.chatState.isLoadingMessage)

  const shouldHideTypingPanel = useSelector(state => {
    const {ids, entities} = state.chatState.messages
    if(ids.length > 0){
      const lastId = ids[ids.length - 1]
      const lastMessage = entities[lastId]
      return (lastMessage?.metadata?.hide_typing === true) && !state.chatState.isLoadingMessage
    }else return false
  })

  const isMobile = useIsMobile()

  const dispatch = useDispatch()

  const {
    register,
    handleSubmit,
    reset,
    setFocus,
  } = useForm({
    resolver: yupResolver(schema),
    reValidateMode: 'onChange'
  })


  const onSubmit = useCallback((data: { message: string }) => {
    if (!isSystemMessageLoading) {
      reset({ message: "" })
      dispatch(sendUserMessage({ id: nanoid(), message: data.message, is_user: true, metadata: undefined, timestamp: Date.now() }))
    }
  }, [isSystemMessageLoading])


  const handleKeyDownOnNameField = useCallback((ev: KeyboardEvent<HTMLTextAreaElement>)=>{
    if(isMobile === false && ev.key == 'Enter' && ev.shiftKey === false){
      ev.preventDefault()
      handleSubmit(onSubmit)()
    }
}, [isMobile, handleSubmit, onSubmit])

  const onTypingViewFocusIn = useCallback((ev: FocusEvent<HTMLTextAreaElement, Element>)=>{
    props.onFocus?.()
  }, [props.onFocus])

  const onTypingViewFocusOut = useCallback((ev: FocusEvent<HTMLTextAreaElement, Element>)=>{
    props.onBlur?.()
  }, [props.onBlur])

  const [t] = useTranslation()

  useEffect(() => {
    setFocus('message')
  }, [setFocus])

  return shouldHideTypingPanel ? null : <>
    <div id="chat-typing-panel" className="sm:fixed sm:z-10 sm:left-4 sm:right-4 sm:bottom-10 lg:left-0 lg:right-0">
      <div className="container relative">
        <form onSubmit={handleSubmit(onSubmit)} className="flex flex-row bg-slate-50 px-3 py-1.5 pl-1.5 sm:rounded-lg shadow-lg">
          {
            isSystemMessageLoading
              ? <div className="text-input text-chat-1 animate-pulse-fast flex-1 mr-2">{t("CHAT.PROCESSING")}</div>
              : <TextareaAutosize {...register("message")} minRows={1} maxRows={5} autoFocus={true} placeholder={t("CHAT.INPUT_PLACEHOLDER")}
                className="chat-type flex-1 mr-2"
                autoComplete="off"
                onFocus={onTypingViewFocusIn}
                onBlur={onTypingViewFocusOut}
                onKeyDown={handleKeyDownOnNameField}
              />
          }
          <button type="submit" className="button-main" disabled={isSystemMessageLoading}>
            {
              isMobile ? <PaperAirplaneIcon className="w-5"/> : <span>{t("LABEL.SEND")}</span>
            }
          </button>

        </form>
      </div>


    </div>
    <div className="bg-background/70 fixed bottom-0 left-10 right-10 h-[50px] collapse sm:visible" /></>
}




const TutorGameOverlay = (props: {
  sessionId: string
  messageId: string
  plan: TutorGamePlan
  question?: string
  onRequestClose: (context: { reason: "quit" | "completed" }) => void
  onSummary: (data: { userMessage: ChatMessage; assistantMessage?: ChatMessage }) => void
}) => {
  useEffect(() => {
    const previousOverflow = document.body.style.overflow
    document.body.style.overflow = "hidden"
    return () => {
      document.body.style.overflow = previousOverflow
    }
  }, [])

  return (
    <div
      role="dialog"
      aria-modal="true"
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/70 px-3 py-6"
    >
      <div className="relative h-full w-full max-h-[92vh] max-w-4xl">
        <button
          type="button"
          onClick={() => props.onRequestClose({ reason: "quit" })}
          className="absolute right-4 top-4 z-10 rounded-full bg-white/90 px-3 py-2 text-sm font-semibold text-slate-600 shadow hover:bg-white hover:text-slate-800 focus:outline-none focus:ring-2 focus:ring-indigo-300"
          aria-label="Close study quest"
        >
          X
        </button>
        <div className="h-full overflow-hidden rounded-3xl bg-white shadow-2xl ring-1 ring-slate-200">
          <TutorGamePage
            embed
            sessionId={props.sessionId}
            messageId={props.messageId}
            plan={props.plan}
            question={props.question}
            onRequestClose={props.onRequestClose}
            onSummary={props.onSummary}
          />
        </div>
      </div>
    </div>
  )
}



const ShareButton = () => {

  const sessionId = useSelector(state => state.chatState.sessionInfo!.sessionId)
  const urlOrigin = useMemo(() => new URL(window.location.href).origin, [])
  const shareURL = useMemo(() => {
    return path.join(urlOrigin, "share", sessionId)
  }, [urlOrigin, sessionId])

  const [t] = useTranslation()

  const onCopy = useCallback((text: string, result: boolean) => {
    enqueueSnackbar(t("CHAT.LINK_COPIED"), {
      autoHideDuration: 1000,
      preventDuplicate: true
    })
  }, [t])

  return <CopyToClipboard text={shareURL} onCopy={onCopy}>
    <button className="button-clear button-tiny button-with-icon opacity-70">
      <ClipboardDocumentIcon className="w-4 mr-1 opacity-70" />
      <span>{t("CHAT.SHARE_LINK")}</span>
    </button></CopyToClipboard>
}

const SessionMessageView = (props: { id: EntityId, isLast: boolean }) => {

  const dispatch = useDispatch()

  const userName = useSelector(state => state.chatState.sessionInfo?.name!)
  const sessionId = useSelector(state => state.chatState.sessionInfo?.sessionId!)

  const turn = useSelector(state => state.chatState.messages.entities[props.id]!)
  const isEmotionSelectionTurn = turn.metadata?.select_emotion === true
  const isTutorModeTurn = turn.metadata?.tutor_mode === true
  const tutorBadge = turn.metadata?.tutor_badge
  const tutorPlan = (turn.metadata?.tutor_game_plan ?? undefined) as TutorGamePlan | undefined
  const showTutorQuest = !turn.is_user && !!tutorPlan && turn.metadata?.tutor_game_ready === true
  const tutorQuestion = turn.metadata?.tutor_assignment_prompt ?? turn.metadata?.assignment_request ?? turn.metadata?.assignment_initial_prompt

  const emotionSelectionResult = useSelector(state => {
    const turn = state.chatState.messages.entities[props.id]!
    const isEmotionSelectionTurn = turn.metadata?.select_emotion === true
    if(!props.isLast && turn.is_user === false && isEmotionSelectionTurn === true){
      const index = state.chatState.messages.ids.indexOf(turn.id)
      const resp = state.chatState.messages.entities[state.chatState.messages.ids[index+1]]
      if(resp?.is_user === true){
        const emotions = [...resp!.message.matchAll(/{key\:\s+\"([a-zA-Z]+)\"}/g)].map(arr => arr[1])
        return emotions.reduce((obj: any, emotion)=>{
          obj[emotion] = true
          return obj
        }, {})
      }else return undefined
    }else return undefined
  })

  const hideMessage = turn.metadata?.hide === true

  const isSystemBusy = useSelector(state => state.chatState.isLoadingMessage)

  const [t] = useTranslation()

  const componentsBelow: Array<JSX.Element> = []
  if (showTutorQuest && tutorPlan) {
    componentsBelow.push(
      <TutorQuestCard key="tutor-quest" plan={tutorPlan} sessionId={sessionId} messageId={turn.id} question={tutorQuestion} />
    )
  }
  if (isEmotionSelectionTurn) {
    componentsBelow.push(
      <EmotionPicker
        key="emotion-picker"
        messageId={props.id}
        disabled={!props.isLast || isSystemBusy === true}
        value={emotionSelectionResult}
      />
    )
  }

  const onDoubleClick = useCallback(()=>{
    if(turn.is_user === false && props.isLast === true){
      if(confirm(t("CHAT.CONFIRM_REGEN_LAST_MESSAGE"))){
        dispatch(regenerateLastSystemMessage())
      }
    }
  }, [turn.is_user, props.isLast, t])

  return hideMessage ? null : <MessageView avatarHash={turn.is_user === true ? userName : "system"} message={turn} onThumbnailDoubleClick={onDoubleClick}
    componentsAboveCallout={
      (!turn.is_user && isTutorModeTurn)
        ? <TutorBadge title={tutorBadge?.title} subtitle={tutorBadge?.subtitle}/>
        : null
    }
    componentsBelowCallout={
      componentsBelow.length > 0 ? <div className="mt-3 flex flex-col gap-3">{componentsBelow}</div> : null
    }/>
}
