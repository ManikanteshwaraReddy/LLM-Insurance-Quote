import React, { useEffect, useRef, useState } from "react";
import { useForm } from "react-hook-form";
import { Bot, Download, Send, User, CheckCircle2, ChevronRight, ShieldCheck, Zap } from "lucide-react";
import { NavLink } from "react-router-dom";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
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

const stages = ["Basic Info", "Health History", "Lifestyle", "Coverage Needs"];

const stageForQuestion = (questionIndex) => {
  if (questionIndex < 5) return 0;
  if (questionIndex < 12) return 1;
  if (questionIndex < 17) return 2;
  return 3;
};

const initialFormData = Object.fromEntries(questions.map(({ key }) => [key, ""]));

const TypingIndicator = () => (
  <div className="flex items-center space-x-1 px-2 py-1" aria-label="Bot is typing">
    <div className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground [animation-delay:-0.3s]"></div>
    <div className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground [animation-delay:-0.15s]"></div>
    <div className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground"></div>
  </div>
);

const Stepper = ({ currentStage }) => (
  <div className="flex items-center justify-between sticky top-0 z-20 bg-background/80 backdrop-blur-xl px-4 md:px-8 py-4 border-b border-border/50">
    {stages.map((stage, index) => {
      const isActive = index === currentStage;
      const isCompleted = index < currentStage;
      return (
        <React.Fragment key={stage}>
          <div className="flex flex-col items-center">
            <div className={`flex h-8 w-8 items-center justify-center rounded-full border-2 text-xs font-semibold transition-all duration-300 ${
              isCompleted ? "bg-primary border-primary text-primary-foreground scale-95" : isActive ? "border-primary text-primary shadow-[0_0_15px_rgba(var(--primary),0.2)]" : "border-muted text-muted-foreground"
            }`}>
              {isCompleted ? <CheckCircle2 className="h-4 w-4" /> : index + 1}
            </div>
            <span className={`mt-2 text-[11px] font-medium hidden sm:block ${isActive || isCompleted ? "text-foreground" : "text-muted-foreground"}`}>{stage}</span>
          </div>
          {index < stages.length - 1 && (
            <div className={`h-[2px] flex-1 mx-2 sm:mx-4 rounded transition-colors duration-500 ${isCompleted ? "bg-primary" : "bg-muted"}`} />
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
      content: "Hi there! I will ask a few short questions, one at a time, to prepare your health insurance estimate. Ready to get started?",
      sender: "bot",
      options: ["Yes, let's begin", "Tell me more first"],
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    },
  ]);
  const [questionIndex, setQuestionIndex] = useState(-1);
  const [showQuote, setShowQuote] = useState(false);
  const [quoteData, setQuoteData] = useState(null);
  const [quoteError, setQuoteError] = useState(null);
  const [isGeneratingQuote, setIsGeneratingQuote] = useState(false);
  const [isTyping, setIsTyping] = useState(false);

  // Auto-scroll logic
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTo({
        top: chatContainerRef.current.scrollHeight,
        behavior: 'smooth'
      });
    }
  }, [messages, isTyping, quoteError, isGeneratingQuote]);

  // Auto-focus logic
  useEffect(() => {
    if (!isTyping && !showQuote && !isGeneratingQuote && inputRef.current) {
      // Small timeout to allow render to settle
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  }, [isTyping, showQuote, isGeneratingQuote]);

  // Merge ref for React Hook Form and manual focus
  const { ref: formInputRef, ...inputProps } = register("message");

  const addBotMessage = (content, options = []) => {
    setMessages((previous) => [...previous, { 
      id: `${Date.now()}-bot`, 
      content, 
      sender: "bot", 
      options,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }]);
  };

  const askQuestion = (index) => {
    setIsTyping(true);
    setTimeout(() => {
      const question = questions[index];
      setQuestionIndex(index);
      setIsTyping(false);
      addBotMessage(question.prompt, question.options || []);
    }, 600 + Math.random() * 300); // slightly faster for better flow
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
        setIsTyping(true);
        setTimeout(() => {
          setIsTyping(false);
          addBotMessage("I will collect your personal details, health background, lifestyle, and coverage preferences. It only takes a few minutes.", ["Okay, let's begin"]);
        }, 800);
        return;
      }
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

    setIsTyping(true);
    setTimeout(() => {
      setIsTyping(false);
      addBotMessage("Thank you. I have everything I need and am generating your personalised quote now.");
      window.setTimeout(generateQuote, 1000);
    }, 800);
  };

  const handleUserMessage = (content) => {
    const answer = content.trim();
    if (!answer) return;
    setMessages((previous) => [...previous, { 
      id: `${Date.now()}-user`, 
      content: answer, 
      sender: "user",
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }]);
    advanceConversation(answer);
  };

  const onSubmit = ({ message }) => {
    handleUserMessage(message || "");
    reset();
  };

  const currentStage = questionIndex < 0 ? 0 : stageForQuestion(questionIndex);
  const currentQuestion = questions[questionIndex];

  return (
    <div className="flex flex-col h-[calc(100vh-64px)] bg-background">
      {!showQuote ? (
        <div className="flex flex-col h-full relative max-w-5xl mx-auto w-full animate-in fade-in duration-500">
          <Stepper currentStage={currentStage} />
          
          <div 
            ref={chatContainerRef} 
            className="flex-1 overflow-y-auto px-4 md:px-8 pt-6 space-y-6 pb-32"
            aria-live="polite" 
            aria-atomic="false"
          >
            {messages.map((message) => (
              <div key={message.id} className={`flex ${message.sender === "user" ? "justify-end" : "justify-start"} animate-in fade-in slide-in-from-bottom-2 duration-300`}>
                <div className={`flex max-w-[90%] md:max-w-[75%] items-end gap-2 ${message.sender === "user" ? "flex-row-reverse" : "flex-row"}`}>
                  
                  {message.sender === "bot" ? (
                    <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary/10 text-primary">
                      <Bot className="h-4 w-4" />
                    </div>
                  ) : (
                    <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-muted text-muted-foreground">
                      <User className="h-4 w-4" />
                    </div>
                  )}

                  <div className="flex flex-col gap-1">
                    <div className={`px-4 py-3 shadow-xs ${
                      message.sender === "user" 
                        ? "bg-primary text-primary-foreground rounded-2xl rounded-br-sm" 
                        : "bg-muted text-foreground rounded-2xl rounded-bl-sm"
                    }`}>
                      <p className="text-[15px] leading-relaxed">{message.content}</p>
                    </div>
                    
                    <span className={`text-[10px] text-muted-foreground px-1 ${message.sender === "user" ? "text-right" : "text-left"}`}>
                      {message.timestamp}
                    </span>

                    {message.options?.length > 0 && (
                      <div className="mt-2 flex flex-wrap gap-2">
                        {message.options.map((option) => (
                          <Button 
                            key={option} 
                            variant="outline" 
                            size="sm" 
                            onClick={() => handleUserMessage(option)} 
                            className="rounded-full bg-card transition-all hover:border-primary hover:text-primary active:scale-95 shadow-xs"
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
            
            {isTyping && (
               <div className="flex justify-start animate-in fade-in duration-300">
                  <div className="flex items-end gap-2">
                    <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary/10 text-primary">
                      <Bot className="h-4 w-4" />
                    </div>
                    <div className="bg-muted rounded-2xl rounded-bl-sm px-4 py-3 w-16">
                      <TypingIndicator />
                    </div>
                  </div>
               </div>
            )}
            
            {isGeneratingQuote && (
              <div className="flex items-center justify-center p-4">
                 <div className="flex items-center gap-3 text-muted-foreground">
                   <div className="h-4 w-4 animate-spin rounded-full border-2 border-primary border-t-transparent"></div>
                   <span className="text-sm font-medium">Crunching the numbers...</span>
                 </div>
              </div>
            )}
            
            {quoteError && (
              <div className="rounded-xl border border-destructive/20 bg-destructive/10 p-4 text-center mx-auto max-w-md">
                <p className="text-sm font-medium text-destructive">{quoteError}</p>
                <Button variant="outline" size="sm" className="mt-3" onClick={generateQuote} disabled={isGeneratingQuote}>Try again</Button>
              </div>
            )}
            
            {/* Spacer to prevent chat from hiding behind the absolute positioned input bar */}
            <div className="h-4 w-full flex-shrink-0"></div>
          </div>

          {/* Floating Command Bar Input */}
          <div className="absolute bottom-4 left-0 right-0 px-4 md:px-8">
            <div className="max-w-3xl mx-auto backdrop-blur-xl bg-background/80 rounded-full border border-border/50 shadow-md p-1 transition-all focus-within:ring-2 focus-within:ring-primary/30 focus-within:shadow-primary/10">
              <form onSubmit={handleSubmit(onSubmit)} className="relative flex items-center w-full">
                <Input 
                  {...inputProps}
                  ref={(e) => {
                    formInputRef(e);
                    inputRef.current = e;
                  }}
                  type={currentQuestion?.inputMode === "numeric" || currentQuestion?.inputMode === "decimal" ? "number" : "text"} 
                  step={currentQuestion?.inputMode === "decimal" ? "0.1" : undefined} 
                  placeholder={isTyping ? "AI is typing..." : "Type your answer here..."} 
                  className="w-full pl-6 pr-12 py-6 rounded-full bg-transparent border-none focus-visible:ring-0 text-[15px] shadow-none" 
                  disabled={isGeneratingQuote || isTyping}
                  autoComplete="off"
                  aria-label="Chat input"
                />
                <Button 
                  type="submit" 
                  size="icon"
                  className="absolute right-1 rounded-full h-10 w-10 bg-primary hover:bg-primary/90 transition-transform active:scale-95 text-primary-foreground" 
                  disabled={isGeneratingQuote || isTyping}
                  aria-label="Send message"
                >
                  <Send className="h-4 w-4 ml-0.5" />
                </Button>
              </form>
            </div>
          </div>
        </div>
      ) : (
        <div className="container mx-auto px-4 py-8 md:py-12 flex-1 animate-in fade-in duration-500">
          <Card className="mx-auto max-w-5xl overflow-hidden border-border bg-card shadow-md animate-in zoom-in-95 duration-700">
            <div className="bg-gradient-to-br from-primary/10 via-background to-background p-8 md:p-12 text-center relative border-b border-border/50">
              <div className="inline-flex items-center justify-center h-16 w-16 rounded-full bg-primary/10 text-primary mb-6 ring-8 ring-primary/5">
                <ShieldCheck className="h-8 w-8" />
              </div>
              <h2 className="mb-3 text-3xl md:text-4xl font-bold tracking-tight text-foreground">Your Premium Health Plan</h2>
              <p className="text-muted-foreground text-lg max-w-xl mx-auto">Tailored coverage based on your profile. Protect what matters most today.</p>
            </div>

            <div className="p-8 md:p-12 bg-muted/20">
              <div className="mb-8 text-center">
                <h3 className="text-xl font-semibold tracking-tight text-foreground">Select your payment schedule</h3>
              </div>
              
              <div className="grid gap-6 md:grid-cols-3">
                {/* Monthly Plan */}
                <div className="rounded-2xl border border-border bg-background p-6 shadow-sm flex flex-col transition-all duration-300 hover:shadow-md hover:border-primary/40 group">
                  <h3 className="text-lg font-medium text-muted-foreground transition-colors group-hover:text-foreground">Monthly</h3>
                  <div className="my-4 text-4xl font-bold tracking-tight text-foreground">
                    ₹{quoteData?.monthlyINR?.toLocaleString("en-IN")}
                  </div>
                  <p className="text-sm text-muted-foreground mb-8 flex-grow">Manageable monthly payments for easier budgeting.</p>
                  <Button variant="outline" className="w-full rounded-xl h-11 active:scale-95 transition-transform">Select Monthly</Button>
                </div>

                {/* Yearly Plan (Recommended) */}
                <div className="rounded-2xl border-2 border-primary bg-background p-6 shadow-md flex flex-col relative transform md:-translate-y-2 z-10 transition-all duration-300 hover:shadow-lg">
                  <div className="absolute -top-4 left-1/2 -translate-x-1/2 rounded-full bg-primary px-4 py-1 text-xs font-bold text-primary-foreground shadow-xs flex items-center gap-1 uppercase tracking-wider">
                    <Zap className="h-3 w-3 fill-current" /> Recommended
                  </div>
                  <h3 className="text-lg font-semibold text-primary">Annually</h3>
                  <div className="my-4 text-4xl font-bold tracking-tight text-foreground">
                    ₹{quoteData?.yearlyINR?.toLocaleString("en-IN")}
                  </div>
                  <p className="text-sm text-muted-foreground mb-8 flex-grow">Best value. Save significantly on premium costs with one payment.</p>
                  <Button className="w-full font-semibold shadow-md rounded-xl h-11 active:scale-95 transition-transform">Select Annually</Button>
                </div>

                {/* Quarterly Plan */}
                <div className="rounded-2xl border border-border bg-background p-6 shadow-sm flex flex-col transition-all duration-300 hover:shadow-md hover:border-primary/40 group">
                  <h3 className="text-lg font-medium text-muted-foreground transition-colors group-hover:text-foreground">Quarterly</h3>
                  <div className="my-4 text-4xl font-bold tracking-tight text-foreground">
                    ₹{quoteData?.quarterlyINR?.toLocaleString("en-IN")}
                  </div>
                  <p className="text-sm text-muted-foreground mb-8 flex-grow">A balanced approach. Pay every three months.</p>
                  <Button variant="outline" className="w-full rounded-xl h-11 active:scale-95 transition-transform">Select Quarterly</Button>
                </div>
              </div>

              <div className="mt-12 flex flex-col sm:flex-row justify-center items-center gap-4">
                <NavLink to="/print" className="w-full sm:w-auto">
                  <Button variant="secondary" className="w-full sm:w-auto px-8 h-12 rounded-xl text-base font-medium active:scale-95 transition-transform shadow-xs hover:shadow-sm">
                    <Download className="mr-2 h-5 w-5" /> Download Full Quote
                  </Button>
                </NavLink>
              </div>
              
              <div className="mt-12 border-t border-border/50 pt-8 text-center">
                <p className="text-sm text-muted-foreground mb-4 max-w-2xl mx-auto">
                  This is an estimated premium based on the details provided. Final coverage amount and terms are subject to medical underwriting and verification by the insurance provider.
                </p>
                <NavLink to="/providers">
                  <Button variant="link" className="text-primary font-medium group">
                    View Insurance Providers <ChevronRight className="h-4 w-4 ml-1 transition-transform group-hover:translate-x-1" />
                  </Button>
                </NavLink>
              </div>
            </div>
          </Card>
        </div>
      )}
    </div>
  );
};

export default FullQuoteChat;
