import {BrowserRouter, Navigate, Route, Routes} from "react-router-dom";
import {ChatPage} from "./app/chat/pages/ChatPage";
import {store} from "./redux/store";
import {Provider} from "react-redux";
import { ChatSharePage } from "./app/chat/pages/ChatShare";
import {SnackbarProvider} from 'notistack';
import { SessionSignInPage } from "./app/chat/pages/SessionSignInPage";
import { TutorGamePage } from "./app/tutor-game/pages/TutorGamePage";

export function App() {
    return <Provider store={store}>
        <SnackbarProvider maxSnack={3}><BrowserRouter>
        <Routes>
            <Route index element={<Navigate to={"signin"}/>}/>
            <Route path={"/signin"} element={<SessionSignInPage/>}/>
            <Route path={"/chat/:sessionId"} element={<ChatPage/>}/>
            <Route path={"/share/:sessionId"} element={<ChatSharePage/>}/>
            <Route path={"/game/:sessionId/:messageId"} element={<TutorGamePage/>}/>
        </Routes>
    </BrowserRouter></SnackbarProvider></Provider>
}
