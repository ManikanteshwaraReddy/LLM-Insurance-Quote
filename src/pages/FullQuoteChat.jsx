import React, { useEffect, useRef, useState } from "react";
import { useForm } from "react-hook-form";
import { Bot, Download, Send, User } from "lucide-react";
import { NavLink } from "react-router-dom";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { getQuote } from "@/lib/api";

const questions = [
  { key: "age", prompt: "To begin, how old are you?", inputMode: "numeric" },
  { key: "gender", prompt: "Which gender should we use for this quote?", options: ["Male", "Female", "Other"] },
  { key: "location", prompt: "Which city do you live in?" },
  { key: "occupation", prompt: "What is your occupation?" },
  { key: "numberOfInsuredMembers", prompt: "How many people should this policy cover?", options: ["1", "2", "3", "4", "5"] },
  { key: "familyDetails", prompt: "Please briefly describe the family members to be covered." },
  { key: "preExistingConditions", prompt: "Do you have any pre-existing health conditions? You can say None if there are none." },
  { key: "pastMedicalHistory", prompt: "Is there any past medical history we should consider?" },
  { key: "familyMedicalHistory", prompt: "Is there any relevant family medical history?" },
  { key: "heightCm", prompt: "What is your height in centimetres?", inputMode: "decimal" },
  { key: "weightKg", prompt: "What is your weight in kilograms?", inputMode: "decimal" },
  { key: "pregnancyStatus", prompt: "Is pregnancy relevant to this quote?", options: ["No", "Yes", "Not applicable"] },
  { key: "smokingTobaccoUse", prompt: "Do you use tobacco or smoke?", options: ["No", "Occasional", "Yes"] },
  { key: "alcoholConsumption", prompt: "How often do you consume alcohol?", options: ["Never", "Occasional", "Regular"] },
  { key: "exerciseFrequency", prompt: "How often do you exercise?", options: ["Sedentary", "1-2 times/week", "3-4 times/week", "Daily"] },
  { key: "lifestyle", prompt: "How would you describe your overall lifestyle?" },
  { key: "coverageNeed", prompt: "What are the most important coverage needs for you?" },
  { key: "planType", prompt: "Would you like an individual or family plan?", options: ["Individual", "Family"] },
  { key: "sumInsured", prompt: "What sum insured would you like in INR?", options: ["300000", "500000", "1000000", "2000000"] },
  { key: "policyTermYears", prompt: "What policy term do you prefer?", options: ["1", "2", "3"] },
  { key: "premiumPaymentMode", prompt: "How would you prefer to pay your premium?", options: ["Monthly", "Quarterly", "Half-Yearly", "Yearly"] },
];

const stages = ["Basic Info", "Health History", "Lifestyle", "Coverage Needs", "Quote Generation"];

const stageForQuestion = (questionIndex) => {
  if (questionIndex < 5) return 0;
  if (questionIndex < 12) return 1;
  if (questionIndex < 17) return 2;
  return 3;
};

const initialFormData = Object.fromEntries(questions.map(({ key }) => [key, ""]));

const FullQuoteChat = () => {
  const { register, handleSubmit, reset } = useForm();
  const chatContainerRef = useRef(null);
  const formDataRef = useRef(initialFormData);
  const [messages, setMessages] = useState([
    {
      id: "welcome",
      content: "Hi there! I will ask a few short questions, one at a time, to prepare your health insurance estimate. Ready to get started?",
      sender: "bot",
      options: ["Yes, let's begin", "Tell me more first"],
    },
  ]);
  const [questionIndex, setQuestionIndex] = useState(-1);
  const [showQuote, setShowQuote] = useState(false);
  const [quoteData, setQuoteData] = useState(null);
  const [quoteError, setQuoteError] = useState(null);
  const [isGeneratingQuote, setIsGeneratingQuote] = useState(false);

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages, quoteError, isGeneratingQuote]);

  const addBotMessage = (content, options = []) => {
    setMessages((previous) => [...previous, { id: `${Date.now()}-bot`, content, sender: "bot", options }]);
  };

  const askQuestion = (index) => {
    const question = questions[index];
    setQuestionIndex(index);
    addBotMessage(question.prompt, question.options || []);
  };

  const buildPayload = () => {
    const data = formDataRef.current;
    const heightCm = Number(data.heightCm);
    const weightKg = Number(data.weightKg);
    const bmi = heightCm > 0 && weightKg > 0 ? Number((weightKg / ((heightCm / 100) ** 2)).toFixed(1)) : undefined;

    return {
      age: Number(data.age),
      medicalHistory: data.pastMedicalHistory,
      gender: data.gender,
      location: data.location,
      occupation: data.occupation,
      numberOfInsuredMembers: Number(data.numberOfInsuredMembers),
      familyDetails: data.familyDetails,
      preExistingConditions: data.preExistingConditions,
      pastMedicalHistory: data.pastMedicalHistory,
      familyMedicalHistory: data.familyMedicalHistory,
      heightCm,
      weightKg,
      bmi,
      pregnancyStatus: data.pregnancyStatus,
      smokingTobaccoUse: data.smokingTobaccoUse,
      alcoholConsumption: data.alcoholConsumption,
      exerciseFrequency: data.exerciseFrequency,
      lifestyle: data.lifestyle,
      coverageNeed: data.coverageNeed,
      planType: data.planType,
      sumInsured: Number(data.sumInsured),
      policyTermYears: Number(data.policyTermYears),
      premiumPaymentMode: data.premiumPaymentMode,
    };
  };

  const generateQuote = async () => {
    try {
      setQuoteError(null);
      setIsGeneratingQuote(true);
      const quote = await getQuote(buildPayload());
      setQuoteData(quote);
      setShowQuote(true);
    } catch (error) {
      setQuoteError(error.message || "Unable to generate a quote. Please try again.");
    } finally {
      setIsGeneratingQuote(false);
    }
  };

  const advanceConversation = (answer) => {
    if (questionIndex === -1) {
      if (answer === "Tell me more first") {
        addBotMessage("I will collect your personal details, health background, lifestyle, and coverage preferences. It only takes a few minutes.", ["Okay, let's begin"]);
        return;
      }
      askQuestion(0);
      return;
    }

    const currentQuestion = questions[questionIndex];
    formDataRef.current = { ...formDataRef.current, [currentQuestion.key]: answer };
    const nextIndex = questionIndex + 1;

    if (nextIndex < questions.length) {
      window.setTimeout(() => askQuestion(nextIndex), 500);
      return;
    }

    addBotMessage("Thank you. I have everything I need and am generating your personalised quote now.");
    window.setTimeout(generateQuote, 500);
  };

  const handleUserMessage = (content) => {
    const answer = content.trim();
    if (!answer) return;
    setMessages((previous) => [...previous, { id: `${Date.now()}-user`, content: answer, sender: "user" }]);
    advanceConversation(answer);
  };

  const onSubmit = ({ message }) => {
    handleUserMessage(message || "");
    reset();
  };

  const completedStages = showQuote ? stages.length : questionIndex < 0 ? 0 : Math.min(stageForQuestion(questionIndex) + 1, stages.length - 1);
  const progress = (completedStages / stages.length) * 100;
  const currentQuestion = questions[questionIndex];

  return (
    <div className="container mx-auto max-w-4xl px-4 py-8">
      <h1 className="mb-8 text-center text-3xl font-bold text-blue-900 dark:text-blue-400">Insurance Quote Chat</h1>

      <div className="mb-8">
        <div className="mb-2 flex justify-between">
          {stages.map((stage, index) => <div key={stage} className={`flex-1 text-center text-xs font-medium ${index < completedStages ? "text-green-600 dark:text-green-400" : "text-gray-500"}`}>{stage}</div>)}
        </div>
        <Progress value={progress} className="h-2" />
      </div>

      {!showQuote ? (
        <>
          <div ref={chatContainerRef} className="mb-4 h-[500px] overflow-y-auto rounded-lg border border-gray-200 bg-gray-50 p-4 dark:border-gray-800 dark:bg-gray-900">
            {messages.map((message) => (
              <div key={message.id} className={`mb-4 flex ${message.sender === "user" ? "justify-end" : "justify-start"}`}>
                <div className="flex max-w-[80%] items-start">
                  {message.sender === "bot" && <div className="mr-2 mt-1 rounded-full bg-blue-100 p-2 dark:bg-blue-900"><Bot className="h-5 w-5 text-blue-700 dark:text-blue-400" /></div>}
                  <div className={`rounded-lg px-4 py-2 ${message.sender === "user" ? "bg-blue-600 text-white dark:bg-blue-700" : "border border-gray-200 bg-white dark:border-gray-700 dark:bg-gray-800"}`}>
                    <p>{message.content}</p>
                    {message.options?.length > 0 && <div className="mt-3 flex flex-wrap gap-2">{message.options.map((option) => <Button key={option} variant="outline" size="sm" onClick={() => handleUserMessage(option)} className="text-xs">{option}</Button>)}</div>}
                  </div>
                  {message.sender === "user" && <div className="ml-2 mt-1 rounded-full bg-green-100 p-2 dark:bg-green-900"><User className="h-5 w-5 text-green-700 dark:text-green-400" /></div>}
                </div>
              </div>
            ))}
          </div>

          {isGeneratingQuote && <p className="mb-4 text-center text-sm text-blue-700 dark:text-blue-400">Contacting the quote service...</p>}
          {quoteError && <div className="mb-4 rounded-lg border border-red-200 bg-red-50 p-3 text-center text-sm text-red-700 dark:border-red-900 dark:bg-red-950 dark:text-red-300"><p>{quoteError}</p><Button variant="outline" size="sm" className="mt-2" onClick={generateQuote} disabled={isGeneratingQuote}>Try again</Button></div>}

          <form onSubmit={handleSubmit(onSubmit)} className="flex gap-2">
            <Input {...register("message")} type={currentQuestion?.inputMode === "numeric" || currentQuestion?.inputMode === "decimal" ? "number" : "text"} step={currentQuestion?.inputMode === "decimal" ? "0.1" : undefined} placeholder="Type your answer..." className="flex-grow" disabled={isGeneratingQuote} />
            <Button type="submit" className="bg-blue-600 hover:bg-blue-700" disabled={isGeneratingQuote}><Send className="h-4 w-4" /></Button>
          </form>
        </>
      ) : (
        <Card className="border-2 border-blue-200 p-6 dark:border-blue-900">
          <div className="mb-6 text-center"><h2 className="mb-2 text-2xl font-bold text-blue-900 dark:text-blue-400">Your Personalised Insurance Quote</h2><p className="text-gray-600 dark:text-gray-400">Based on your health profile and coverage preferences</p></div>
          <div className="mb-6 rounded-lg bg-blue-50 p-4 dark:bg-blue-950">
            <h3 className="mb-3 text-xl font-semibold text-blue-900 dark:text-blue-400">Estimated Health Insurance Premium</h3>
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <div><p><strong>Selected payment amount:</strong> INR {quoteData?.totalPayableINR?.toLocaleString("en-IN")}</p><p><strong>Yearly:</strong> INR {quoteData?.yearlyINR?.toLocaleString("en-IN")}</p><p><strong>Half-yearly:</strong> INR {quoteData?.halfYearlyINR?.toLocaleString("en-IN")}</p></div>
              <div><p><strong>Quarterly:</strong> INR {quoteData?.quarterlyINR?.toLocaleString("en-IN")}</p><p><strong>Monthly:</strong> INR {quoteData?.monthlyINR?.toLocaleString("en-IN")}</p></div>
            </div>
          </div>
          <div className="flex justify-center"><NavLink to="/print"><Button className="bg-blue-600 hover:bg-blue-700"><Download className="mr-2 h-4 w-4" /> Download Quote</Button></NavLink></div>
          <div className="mt-8 border-t border-gray-200 pt-6 text-center dark:border-gray-800"><p className="mb-4 text-gray-600 dark:text-gray-400">This is an estimated premium. Confirm final coverage and terms with an insurance provider.</p><NavLink to="/providers"><Button variant="link" className="text-blue-700 dark:text-blue-400">View Insurance Providers</Button></NavLink></div>
        </Card>
      )}
    </div>
  );
};

export default FullQuoteChat;
