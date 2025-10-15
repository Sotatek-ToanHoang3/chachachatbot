import { AcademicCapIcon } from "@heroicons/react/24/solid"
import chachaImage from "../../../CHACHA.png"

export const TutorBadge = (props: { title?: string, subtitle?: string }) => {
  const titleText = props.title ?? "Study Buddy Mode"
  const subtitleText = props.subtitle ?? "Let's work through the assignment together!"

  return (
    <div className="flex items-center gap-3 mb-2 rounded-xl bg-indigo-50/80 px-3 py-2 shadow-sm">
      <img
        src={chachaImage}
        alt="ChaCha study buddy"
        className="w-12 h-12 rounded-full border-2 border-indigo-200 object-cover shadow-sm"
      />
      <div className="text-sm leading-snug text-indigo-950">
        <div className="flex items-center font-semibold text-indigo-700">
          <AcademicCapIcon className="w-4 h-4 mr-1" />
          <span>{titleText}</span>
        </div>
        <p className="text-indigo-600 mt-0.5">{subtitleText}</p>
      </div>
    </div>
  )
}
