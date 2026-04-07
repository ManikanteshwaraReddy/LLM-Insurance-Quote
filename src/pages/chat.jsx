import React, { useState, useRef, useEffect } from "react";
import { useForm } from "react-hook-form";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { Send, Download, Bot, User } from "lucide-react";
import { Card } from "@/components/ui/card";
import { NavLink } from "react-router-dom";

const ChatPage = () => {
  const { register, handleSubmit, reset } = useForm();
  const chatContainerRef = useRef(null);
  const [messages, setMessages] = useState([
    {
      id: "welcome",
      content:
        "Hi there! I'm your insurance assistant. I'll help you find the perfect insurance plan based on your health history and needs. Ready to get started?",
      sender: "bot",
      options: ["Yes, let's begin", "Tell me more first"],
    },
  ]);
  const [showQuote, setShowQuote] = useState(false);
  const [currentStage, setCurrentStage] = useState(0);
  const [introCompleted, setIntroCompleted] = useState(false);
  const [currentQuestionKey, setCurrentQuestionKey] = useState(null);
  const [stages, setStages] = useState([
    { name: "Basic Info", completed: false },
    { name: "Health History", completed: false },
    { name: "Lifestyle", completed: false },
    { name: "Coverage Needs", completed: false },
    { name: "Quote Generation", completed: false },
  ]);
  const [formData, setFormData] = useState({
    age: "",
    medicalHistory: "",
    lifestyle: "",
    coverageNeed: "",
  });
  
  // Add state for quote data
  const [quoteData, setQuoteData] = useState(null);

  const getQuotation = async () => {
    console.log("Generating quotation with data:", formData);
    
    try {
      // For now, return a mock quote - replace with actual API call
      // TODO: Replace with actual backend integration
      const mockQuote = {
        quote_id: `quote_${Date.now()}`,
        plan: {
          name: "Comprehensive Health Plus",
          monthly_premium: 285.00,
          annual_deductible: 1500,
          out_of_pocket_max: 5000,
          coverage_type: "comprehensive",
          network_type: "PPO"
        },
        benefits: [
          "Primary care visits: $25 copay",
          "Specialist visits: $50 copay", 
          "Emergency room: $250 copay",
          "Hospital stay: 20% coinsurance after deductible",
          "Mental health services: $25 copay",
          "Prescription drugs: Tier 1-3 coverage"
        ],
        recommendations: [
          "This plan offers comprehensive coverage suitable for your profile",
          "Consider the moderate deductible with good out-of-pocket protection"
        ],
        confidence_score: 0.85
      };
      
      // Simulate API call delay
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      console.log("Generated quote:", mockQuote);
      return mockQuote;
      
    } catch (error) {
      console.error("Error generating quote:", error);
      // Return fallback quote on error
      return {
        quote_id: `fallback_${Date.now()}`,
        plan: {
          name: "Standard Health Plan",
          monthly_premium: 250.00,
          annual_deductible: 2000,
          out_of_pocket_max: 6000,
          coverage_type: "standard",
          network_type: "PPO"
        },
        benefits: [
          "Primary care visits: $30 copay",
          "Preventive care: Covered 100%",
          "Emergency room: $300 copay"
        ],
        recommendations: [
          "This is a fallback quote - please try again for personalized results"
        ],
        confidence_score: 0.6
      };
    }
  };
  
  const updateStageCompletion = (stageIndex) => {
    setStages((prevStages) =>
      prevStages.map((stage, index) =>
        index === stageIndex ? { ...stage, completed: true } : stage
      )
    );
  };

  const calculateProgress = () => {
    const completedStages = stages.filter((stage) => stage.completed).length;
    return (completedStages / stages.length) * 100;
  };

  const sendBotMessage = (content, options = []) => {
    const botMessage = {
      id: Date.now().toString(),
      content,
      sender: "bot",
      options,
    };
    setMessages((prev) => [...prev, botMessage]);
  };

  const handleUserMessage = (userContent) => {
    const userMessage = {
      id: Date.now().toString(),
      content: userContent,
      sender: "user",
    };
    setMessages((prev) => [...prev, userMessage]);

    if (currentQuestionKey) {
      setFormData((prevData) => ({
        ...prevData,
        [currentQuestionKey]: userContent,
      }));
      setCurrentQuestionKey(null);
    }

    processUserResponse(userContent);
  };

  const processUserResponse = (userContent) => {
    setTimeout(() => {
      if (!introCompleted) {
        if (userContent === "Yes, let's begin") {
          setIntroCompleted(true);
          sendBotMessage(
            "Great! Let's start with some basic information. What's your age?"
          );
          setCurrentQuestionKey("age");
          setCurrentStage(1);
          updateStageCompletion(0);
        } else if (userContent === "Tell me more first") {
          sendBotMessage(
            "I'm here to guide you through a few simple questions about your health and lifestyle. Based on your answers, I'll recommend the best insurance plans for you!",
            ["Okay, let's begin"]
          );
        } else if (userContent === "Okay, let's begin") {
          setIntroCompleted(true);
          sendBotMessage(
            "Awesome! Let's start with your basic information. What's your age?"
          );
          setCurrentQuestionKey("age");
          setCurrentStage(1);
          updateStageCompletion(0);
        }
      } else {
        if (currentStage === 1) {
          sendBotMessage(
            "Thanks! Do you have any pre-existing medical conditions?",
            []
          );
          setCurrentQuestionKey("medicalHistory");
          setCurrentStage(2);
          updateStageCompletion(1);
        } else if (currentStage === 2) {
          sendBotMessage("How would you describe your lifestyle?", [
            "Sedentary",
            "Moderately active",
            "Very active",
          ]);
          setCurrentQuestionKey("lifestyle");
          setCurrentStage(3);
          updateStageCompletion(2);
        } else if (currentStage === 3) {
          sendBotMessage("What type of coverage are you looking for?", [
            "Basic health insurance",
            "Comprehensive coverage",
            "Specialized coverage",
          ]);
          setCurrentQuestionKey("coverageNeed");
          setCurrentStage(4);
          updateStageCompletion(3);
        } else if (currentStage === 4) {
          sendBotMessage(
            "Thank you! I'm generating your personalized insurance quote..."
          );
          updateStageCompletion(4);
          setTimeout(async () => {
            const quote = await getQuotation();
            setQuoteData(quote);
            setShowQuote(true);
          }, 2000);
        }
      }
    }, 800);
  };

  const onSubmit = (data) => {
    if (data.message.trim()) {
      handleUserMessage(data.message);
      reset();
    }
  };

  const handleOptionClick = (option) => {
    handleUserMessage(option);
  };
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages]);
  
  return (
    <div className="container mx-auto max-w-4xl px-4 py-8">
      <h1 className="text-3xl font-bold text-center text-blue-900 mb-8 dark:text-blue-400">
        Insurance Quote Chat
      </h1>

      <div className="mb-8">
        <div className="flex justify-between mb-2">
          {stages.map((stage, index) => (
            <div key={index} className="text-xs text-center flex-1">
              <div
                className={`font-medium ${
                  stage.completed
                    ? "text-green-600 dark:text-green-400"
                    : "text-gray-500"
                }`}
              >
                {stage.name}
              </div>
            </div>
          ))}
        </div>
        <Progress value={calculateProgress()} className="h-2" />
      </div>

      {!showQuote ? (
        <>
          <div 
          ref={chatContainerRef}
          className="bg-gray-50 rounded-lg p-4 mb-4 h-[500px] overflow-y-auto border border-gray-200 dark:bg-gray-900 dark:border-gray-800">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`mb-4 flex ${
                  message.sender === "user" ? "justify-end" : "justify-start"
                }`}
              >
                <div className="flex items-start max-w-[80%]">
                  {message.sender === "bot" && (
                    <div className="mr-2 mt-1 bg-blue-100 rounded-full p-2 dark:bg-blue-900">
                      <Bot className="h-5 w-5 text-blue-700 dark:text-blue-400" />
                    </div>
                  )}
                  <div
                    className={`rounded-lg px-4 py-2 ${
                      message.sender === "user"
                        ? "bg-blue-600 text-white dark:bg-blue-700"
                        : "bg-white border border-gray-200 dark:bg-gray-800 dark:border-gray-700"
                    }`}
                  >
                    <p>{message.content}</p>

                    {message.options && (
                      <div className="mt-3 flex flex-wrap gap-2">
                        {message.options.map((option) => (
                          <Button
                            key={option}
                            variant="outline"
                            size="sm"
                            onClick={() => handleOptionClick(option)}
                            className="text-xs"
                          >
                            {option}
                          </Button>
                        ))}
                      </div>
                    )}
                  </div>
                  {message.sender === "user" && (
                    <div className="ml-2 mt-1 bg-green-100 rounded-full p-2 dark:bg-green-900">
                      <User className="h-5 w-5 text-green-700 dark:text-green-400" />
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>

          <form onSubmit={handleSubmit(onSubmit)} className="flex gap-2">
            <Input
              {...register("message")}
              placeholder="Type your message..."
              className="flex-grow"
            />
            <Button type="submit" className="bg-blue-600 hover:bg-blue-700">
              <Send className="h-4 w-4" />
            </Button>
          </form>
        </>
      ) : (
        <Card className="p-6 border-2 border-blue-200 dark:border-blue-900">
          <div className="text-center mb-6">
            <h2 className="text-2xl font-bold text-blue-900 mb-2 dark:text-blue-400">
              Your Personalized Insurance Quote
            </h2>
            <p className="text-gray-600 dark:text-gray-400">
              Based on your health profile and needs
            </p>
          </div>

          <div className="bg-blue-50 p-4 rounded-lg mb-6 dark:bg-blue-950">
            <h3 className="text-xl font-semibold text-blue-900 mb-3 dark:text-blue-400">
              Recommended Plan: {quoteData?.plan?.name || "Comprehensive Health Plus"}
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <p>
                  <strong>Monthly Premium:</strong> ${quoteData?.plan?.monthly_premium || "285.00"}
                </p>
                <p>
                  <strong>Annual Deductible:</strong> ${quoteData?.plan?.annual_deductible || "1,500"}
                </p>
                <p>
                  <strong>Out-of-pocket Maximum:</strong> ${quoteData?.plan?.out_of_pocket_max || "5,000"}
                </p>
              </div>
              <div>
                <p>
                  <strong>Coverage Type:</strong> {quoteData?.plan?.coverage_type || "Comprehensive"}
                </p>
                <p>
                  <strong>Network Type:</strong> {quoteData?.plan?.network_type || "PPO"}
                </p>
                <p>
                  <strong>Quote ID:</strong> {quoteData?.quote_id || "N/A"}
                </p>
              </div>
            </div>
            {quoteData?.confidence_score && (
              <div className="mt-3 text-sm text-gray-600 dark:text-gray-400">
                <strong>Confidence Score:</strong> {Math.round(quoteData.confidence_score * 100)}%
              </div>
            )}
          </div>

          <div className="mb-6">
            <h3 className="text-lg font-semibold text-blue-900 mb-3 dark:text-blue-400">
              Plan Benefits
            </h3>
            <ul className="list-disc pl-5 space-y-1 text-gray-700 dark:text-gray-300">
              {quoteData?.benefits?.map((benefit, index) => (
                <li key={index}>{benefit}</li>
              )) || (
                <>
                  <li>Primary care visits: $25 copay</li>
                  <li>Specialist visits: $50 copay</li>
                  <li>Emergency room: $250 copay (waived if admitted)</li>
                  <li>Hospital stay: 20% coinsurance after deductible</li>
                  <li>Mental health services: $25 copay</li>
                  <li>Preventive care: Covered 100%</li>
                </>
              )}
            </ul>
          </div>

          {quoteData?.recommendations && quoteData.recommendations.length > 0 && (
            <div className="mb-6">
              <h3 className="text-lg font-semibold text-blue-900 mb-3 dark:text-blue-400">
                Recommendations
              </h3>
              <ul className="list-disc pl-5 space-y-1 text-gray-700 dark:text-gray-300">
                {quoteData.recommendations.map((rec, index) => (
                  <li key={index}>{rec}</li>
                ))}
              </ul>
            </div>
          )}

          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <NavLink to={'/print'}>
            <Button className="bg-blue-600 hover:bg-blue-700">
              <Download className="mr-2 h-4 w-4" /> Download Quote
            </Button>
            </NavLink>
          </div>

          <div className="mt-8 pt-6 border-t border-gray-200 text-center dark:border-gray-800">
            <p className="text-gray-600 mb-4 dark:text-gray-400">
              This quote can be used for negotiations with insurance brokers.
            </p>
            <NavLink to="/providers">
              <Button
                variant="link"
                className="text-blue-700 dark:text-blue-400"
              >
                View Insurance Providers →
              </Button>
            </NavLink>
          </div>
        </Card>
      )}
    </div>
  );
};

export default ChatPage;
