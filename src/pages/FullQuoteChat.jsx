import React, { useEffect, useRef, useState } from "react";
import { useForm } from "react-hook-form";
import { Download, Send, CheckCircle2, ChevronRight, ShieldCheck } from "lucide-react";
import { NavLink } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Alert } from "@/components/ui/alert";
import { Spinner } from "@/components/ui/spinner";
import { getQuote } from "@/lib/api";

const questions = [
  { key: "age", prompt: "How old are you?", inputMode: "numeric" },
  { key: "gender", prompt: "Which gender should we use for this quote?", options: ["Male", "Female", "Other"] },
  { key: "location", prompt: "Which city do you live in?" },
  { key: "occupation", prompt: "What is your occupation?" },
  { key: "numberOfInsuredMembers", prompt: "How many people should this policy cover?", options: ["1", "2", "3", "4", "5"] },
  { key: "familyDetails", prompt: "Briefly describe the family members to be covered." },
  { key: "preExistingConditions", prompt: "Do you have any pre-existing health conditions? (Enter None if inapplicable)" },
  { key: "pastMedicalHistory", prompt: "Is there any past medical history we should consider?" },
  { key: "familyMedicalHistory", prompt: "Is there any relevant family medical history?" },
  { key: "heightCm", prompt: "What is your height in centimetres?", inputMode: "decimal" },
  { key: "weightKg", prompt: "What is your weight in kilograms?", inputMode: "decimal" },
  { key: "pregnancyStatus", prompt: "Is pregnancy coverage relevant to this quote?", options: ["No", "Yes", "Not applicable"] },
  { key: "smokingTobaccoUse", prompt: "Do you use tobacco or smoke?", options: ["No", "Occasional", "Yes"] },
  { key: "alcoholConsumption", prompt: "How often do you consume alcohol?", options: ["Never", "Occasional", "Regular"] },
  { key: "exerciseFrequency", prompt: "How often do you exercise?", options: ["Sedentary", "1-2 times/week", "3-4 times/week", "Daily"] },
  { key: "lifestyle", prompt: "How would you describe your overall lifestyle?" },
  { key: "coverageNeed", prompt: "What are your primary coverage needs?" },
  { key: "planType", prompt: "Would you like an individual or family plan?", options: ["Individual", "Family"] },
  { key: "sumInsured", prompt: "What sum insured would you like in INR?", options: ["300000", "500000", "1000000", "2000000"] },
  { key: "policyTermYears", prompt: "What policy term do you prefer?", options: ["1", "2", "3"] },
  { key: "premiumPaymentMode", prompt: "How would you prefer to pay your premium?", options: ["Monthly", "Quarterly", "Half-Yearly", "Yearly"] },
];

const stages = ["Basic Info", "Health History", "Lifestyle", "Coverage Needs"];

const stageForQuestion = (questionIndex) => {
  if (questionIndex < 5) return 0;
  if (questionIndex < 12) return 1;
  if (questionIndex < 17) return 2;
  return 3;
};

const initialFormData = Object.fromEntries(questions.map(({ key }) => [key, ""]));

const Stepper = ({ currentStage }) => (
  <div className="sticky top-header z-20 flex items-center justify-between border-b border-border bg-card px-4 py-3 md:px-8">
    {stages.map((stage, index) => {
      const isActive = index === currentStage;
      const isCompleted = index < currentStage;
      return (
        <React.Fragment key={stage}>
          <div className="flex items-center gap-2">
            <div
              aria-current={isActive ? "step" : undefined}
              className={`flex h-6 w-6 items-center justify-center rounded-full text-caption font-bold border ${
                isCompleted
                  ? "border-primary bg-primary text-primary-foreground"
                  : isActive
                    ? "border-primary text-primary"
                    : "border-border text-text-tertiary"
              }`}
            >
              {isCompleted ? <CheckCircle2 className="h-3.5 w-3.5" aria-hidden="true" /> : index + 1}
            </div>
            <span className={`hidden text-small font-medium sm:inline ${isActive || isCompleted ? "text-text-primary" : "text-text-tertiary"}`}>
              {stage}
            </span>
          </div>
          {index < stages.length - 1 && (
            <div
              aria-hidden="true"
              className={`mx-2 h-px flex-1 ${isCompleted ? "bg-primary" : "bg-border"}`}
            />
          )}
        </React.Fragment>
      );
    })}
  </div>
);

const FullQuoteChat = () => {
  const { register, handleSubmit, reset } = useForm();
  const chatContainerRef = useRef(null);
  const inputRef = useRef(null);
  const formDataRef = useRef(initialFormData);

  const [messages, setMessages] = useState([
    {
      id: "welcome",
      content: "Welcome to the health quote assistant. Answer each question to receive your personalized premium estimate.",
      sender: "bot",
      options: ["Start Assessment"],
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    },
  ]);
  const [questionIndex, setQuestionIndex] = useState(-1);
  const [showQuote, setShowQuote] = useState(false);
  const [quoteData, setQuoteData] = useState(null);
  const [quoteError, setQuoteError] = useState(null);
  const [isGeneratingQuote, setIsGeneratingQuote] = useState(false);

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTo({
        top: chatContainerRef.current.scrollHeight,
        behavior: "smooth",
      });
    }
  }, [messages, quoteError, isGeneratingQuote]);

  useEffect(() => {
    if (!showQuote && !isGeneratingQuote && inputRef.current) {
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  }, [showQuote, isGeneratingQuote]);

  const { ref: formInputRef, ...inputProps } = register("message");

  const addBotMessage = (content, options = []) => {
    setMessages((previous) => [
      ...previous,
      {
        id: `${Date.now()}-bot`,
        content,
        sender: "bot",
        options,
        timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      },
    ]);
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
      askQuestion(0);
      return;
    }

    const currentQuestion = questions[questionIndex];
    formDataRef.current = { ...formDataRef.current, [currentQuestion.key]: answer };
    const nextIndex = questionIndex + 1;

    if (nextIndex < questions.length) {
      askQuestion(nextIndex);
      return;
    }

    addBotMessage("Thank you. Computing policy estimate...");
    generateQuote();
  };

  const handleUserMessage = (content) => {
    const answer = content.trim();
    if (!answer) return;
    setMessages((previous) => [
      ...previous,
      {
        id: `${Date.now()}-user`,
        content: answer,
        sender: "user",
        timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      },
    ]);
    advanceConversation(answer);
  };

  const onSubmit = ({ message }) => {
    handleUserMessage(message || "");
    reset();
  };

  const currentStage = questionIndex < 0 ? 0 : stageForQuestion(questionIndex);
  const currentQuestion = questions[questionIndex];

  return (
    <div className="flex h-app-viewport flex-col bg-background">
      {!showQuote ? (
        <div className="relative mx-auto flex h-full w-full max-w-3xl flex-col">
          <Stepper currentStage={currentStage} />

          <div
            ref={chatContainerRef}
            className="flex-1 space-y-4 overflow-y-auto px-4 pb-24 pt-6 md:px-6"
            role="log"
            aria-live="polite"
          >
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex ${message.sender === "user" ? "justify-end" : "justify-start"}`}
              >
                <div className={`flex max-w-[85%] items-start gap-2.5 ${message.sender === "user" ? "flex-row-reverse" : "flex-row"}`}>
                  <div
                    className={`rounded-lg px-4 py-3 text-small leading-relaxed border ${
                      message.sender === "user"
                        ? "bg-primary text-primary-foreground border-primary"
                        : "bg-card text-text-primary border-border"
                    }`}
                  >
                    <p>{message.content}</p>
                    {message.options?.length > 0 && (
                      <div className="mt-3 flex flex-wrap gap-2">
                        {message.options.map((option) => (
                          <Button
                            key={option}
                            variant="outline"
                            size="sm"
                            onClick={() => handleUserMessage(option)}
                          >
                            {option}
                          </Button>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}

            {isGeneratingQuote && (
              <div className="flex items-center justify-center py-4">
                <div className="flex items-center gap-2 text-text-secondary text-small">
                  <Spinner className="h-4 w-4 text-primary" />
                  <span>Calculating estimate...</span>
                </div>
              </div>
            )}

            {quoteError && (
              <Alert type="error" message={quoteError} className="mx-auto max-w-md">
                <div className="pt-2 text-center">
                  <Button variant="outline" size="sm" onClick={generateQuote} disabled={isGeneratingQuote}>
                    Retry
                  </Button>
                </div>
              </Alert>
            )}
          </div>

          {/* Input Bar */}
          <div className="border-t border-border bg-card p-3">
            <form onSubmit={handleSubmit(onSubmit)} className="flex items-center gap-2 max-w-3xl mx-auto">
              <Input
                {...inputProps}
                ref={(e) => {
                  formInputRef(e);
                  inputRef.current = e;
                }}
                type={currentQuestion?.inputMode === "numeric" || currentQuestion?.inputMode === "decimal" ? "number" : "text"}
                step={currentQuestion?.inputMode === "decimal" ? "0.1" : undefined}
                placeholder="Type your answer..."
                className="flex-1"
                disabled={isGeneratingQuote}
                autoComplete="off"
                aria-label="Form answer input"
              />
              <Button
                type="submit"
                size="icon"
                disabled={isGeneratingQuote}
                aria-label="Send response"
              >
                <Send className="h-4 w-4" />
              </Button>
            </form>
          </div>
        </div>
      ) : (
        <div className="container mx-auto max-w-4xl px-4 py-8 md:py-12">
          <div className="rounded-xl border border-border bg-card p-6 md:p-10">
            <div className="border-b border-border pb-6 mb-8 text-center sm:text-left">
              <div className="inline-flex items-center gap-2 text-caption font-semibold text-success uppercase tracking-wider mb-1">
                <ShieldCheck className="h-4 w-4" /> Assessment Complete
              </div>
              <h2 className="text-h1 font-bold text-text-primary tracking-tight">Estimated Policy Premiums</h2>
              <p className="mt-1 text-small text-text-secondary">
                Calculated based on your age, medical history, location, and coverage choices.
              </p>
            </div>

            <div className="grid gap-6 md:grid-cols-3 mb-8">
              {/* Monthly */}
              <div className="rounded-lg border border-border p-5 bg-background flex flex-col justify-between">
                <div>
                  <span className="text-label font-semibold text-text-secondary uppercase">Monthly</span>
                  <div className="my-2 text-h1 font-bold text-text-primary">
                    ₹{quoteData?.monthlyINR?.toLocaleString("en-IN")}
                  </div>
                  <p className="text-caption text-text-secondary">Monthly payment schedule.</p>
                </div>
                <Button variant="outline" className="mt-6 w-full">Select Monthly</Button>
              </div>

              {/* Annual */}
              <div className="rounded-lg border-2 border-primary p-5 bg-card flex flex-col justify-between">
                <div>
                  <span className="text-label font-semibold text-primary uppercase">Annual</span>
                  <div className="my-2 text-h1 font-bold text-text-primary">
                    ₹{quoteData?.yearlyINR?.toLocaleString("en-IN")}
                  </div>
                  <p className="text-caption text-text-secondary">Single annual payment.</p>
                </div>
                <Button className="mt-6 w-full">Select Annual</Button>
              </div>

              {/* Quarterly */}
              <div className="rounded-lg border border-border p-5 bg-background flex flex-col justify-between">
                <div>
                  <span className="text-label font-semibold text-text-secondary uppercase">Quarterly</span>
                  <div className="my-2 text-h1 font-bold text-text-primary">
                    ₹{quoteData?.quarterlyINR?.toLocaleString("en-IN")}
                  </div>
                  <p className="text-caption text-text-secondary">Quarterly payment schedule.</p>
                </div>
                <Button variant="outline" className="mt-6 w-full">Select Quarterly</Button>
              </div>
            </div>

            <div className="flex flex-col sm:flex-row items-center justify-between gap-4 pt-6 border-t border-border">
              <NavLink to="/print">
                <Button variant="secondary" className="gap-2">
                  <Download className="h-4 w-4" /> Download Quote Summary
                </Button>
              </NavLink>

              <NavLink to="/providers">
                <Button variant="link" className="text-caption font-semibold">
                  Browse Carrier Partners <ChevronRight className="h-3.5 w-3.5" />
                </Button>
              </NavLink>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default FullQuoteChat;
