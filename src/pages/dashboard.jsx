import React from 'react'
import { NavLink } from 'react-router'
import { Button } from "@/components/ui/button"
import { ArrowRight, Shield, Clock, MessageSquare } from "lucide-react"

const Dashboard = () => {
  return (
    <div className="flex flex-col min-h-screen select-none">
      <section className="bg-gradient-to-b from-blue-50 to-white py-20 px-4 md:px-6 lg:px-8 border-b ">
        <div className="container mx-auto max-w-6xl">
          <div className="flex flex-col md:flex-row items-center gap-12">
            <div className="flex-1 space-y-6">
              <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold text-blue-900 leading-tight">
                Get insurance quotes tailored to your health history, instantly!
              </h1>
              <p className="text-lg md:text-xl text-blue-700 max-w-2xl">
                Our AI-powered platform analyzes your unique health profile to find the perfect insurance coverage for
                your needs.
              </p>
              <div className="pt-4">
                <NavLink to="/chat">
                  <Button
                    size="lg"
                    className="cursor-pointer bg-green-600 hover:bg-green-700 text-white rounded-full px-8 py-6 text-lg font-medium"
                  >
                    Start Chat <ArrowRight className="ml-2 h-5 w-5" />
                  </Button>
                </NavLink>
              </div>

              <div className="flex flex-wrap gap-6 pt-8">
                <div className="flex items-center gap-2">
                  <Shield className="h-5 w-5 text-green-600" />
                  <span className="text-gray-700">Secure & Confidential</span>
                </div>
                <div className="flex items-center gap-2">
                  <Clock className="h-5 w-5 text-green-600" />
                  <span className="text-gray-700">Quick 5-Minute Process</span>
                </div>
                <div className="flex items-center gap-2">
                  <MessageSquare className="h-5 w-5 text-green-600" />
                  <span className="text-gray-700">Personalized Recommendations</span>
                </div>
              </div>
            </div>
            <div className="hidden flex-1 justify-center md:block">
              <div className="relative w-full max-w-md">
                <div className="absolute -z-10 w-72 h-72 bg-blue-100 rounded-full blur-3xl opacity-70 top-0 -right-10"></div>
                <div className="absolute -z-10 w-72 h-72 bg-green-100 rounded-full blur-3xl opacity-70 -bottom-10 -left-10"></div>
                <img
                  src="/landing-image.png?height=400&width=400"
                  alt="Insurance Illustration"
                  className="w-full h-auto relative z-10"
                />
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="py-20 px-4 md:px-6 lg:px-8 bg-white">
        <div className="container mx-auto max-w-6xl">
          <h2 className="text-3xl font-bold text-center text-blue-900 mb-16">How It Works</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-10">
            <div className="bg-blue-50 rounded-xl p-8 text-center shadow-sm hover:shadow-md transition-shadow">
              <div className="bg-blue-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-6">
                <MessageSquare className="h-8 w-8 text-blue-700" />
              </div>
              <h3 className="text-xl font-semibold text-blue-900 mb-3">Chat with our AI</h3>
              <p className="text-gray-700">Answer simple questions about your health history and insurance needs.</p>
            </div>

            <div className="bg-green-50 rounded-xl p-8 text-center shadow-sm hover:shadow-md transition-shadow">
              <div className="bg-green-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-6">
                <Shield className="h-8 w-8 text-green-700" />
              </div>
              <h3 className="text-xl font-semibold text-green-900 mb-3">Get Personalized Quotes</h3>
              <p className="text-gray-700">Receive tailored insurance quotes based on your unique profile.</p>
            </div>

            <div className="bg-blue-50 rounded-xl p-8 text-center shadow-sm hover:shadow-md transition-shadow">
              <div className="bg-blue-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-6">
                <ArrowRight className="h-8 w-8 text-blue-700" />
              </div>
              <h3 className="text-xl font-semibold text-blue-900 mb-3">Connect with Providers</h3>
              <p className="text-gray-700">
                Use your quote to negotiate with insurance brokers or contact providers directly.
              </p>
            </div>
          </div>

          <div className="mt-16 text-center">
            <NavLink to="/providers">
              <Button variant="outline" className="cursor-pointer rounded-full border-blue-300 text-blue-700 hover:bg-blue-50">
                View Insurance Providers
              </Button>
            </NavLink>
          </div>
        </div>
      </section>
    </div>
  )
}

export default Dashboard
