import React, { useEffect } from "react";
import { NavLink } from "react-router";
import { Button } from "@/components/ui/button";
import { ArrowLeft } from "lucide-react";

const TermsOfService = () => {
  useEffect(() => {
    document.title = "Terms of Service";
  }, []);

  return (
    <div className="container mx-auto max-w-4xl px-4 py-12">
      <div className="mb-8">
        <NavLink to={"/"}>
          <Button
            variant="ghost"
            className="pl-0 text-blue-700 dark:text-blue-400"
          >
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Home
          </Button>
        </NavLink>
      </div>

      <div className="bg-white rounded-lg shadow-sm border border-gray-100 p-8 dark:bg-gray-900 dark:border-gray-800">
        <h1 className="text-3xl font-bold text-blue-900 mb-6 dark:text-blue-400">
          Terms of Service
        </h1>
        <p className="text-gray-600 mb-8 dark:text-gray-400">
          Last Updated: May 8, 2025
        </p>

        <div className="space-y-8">
          <section>
            <h2 className="text-xl font-semibold text-blue-800 mb-3 dark:text-blue-300">
              1. Introduction
            </h2>
            <p className="text-gray-700 mb-3 dark:text-gray-300">
              Welcome to SmartHealthQuote. These terms and conditions outline
              the rules and regulations for the use of our website and services.
            </p>
            <p className="text-gray-700 dark:text-gray-300">
              By accessing this website, we assume you accept these terms and
              conditions in full. Do not continue to use SmartHealthQuote if
              you do not accept all of the terms and conditions stated on this
              page.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-semibold text-blue-800 mb-3 dark:text-blue-300">
              2. License to Use
            </h2>
            <p className="text-gray-700 mb-3 dark:text-gray-300">
              Unless otherwise stated, SmartHealthQuote and/or its licensors
              own the intellectual property rights for all material on this
              website. All intellectual property rights are reserved.
            </p>
            <p className="text-gray-700 dark:text-gray-300">
              You may view and/or print pages from the website for your own
              personal use subject to restrictions set in these terms and
              conditions.
            </p>
            <p className="text-gray-700 mt-3 dark:text-gray-300">
              You must not:
            </p>
            <ul className="list-disc pl-6 space-y-2 text-gray-700 dark:text-gray-300">
              <li>Republish material from this website</li>
              <li>Sell, rent, or sub-license material from this website</li>
              <li>Reproduce, duplicate, or copy material from this website</li>
              <li>
                Redistribute content from SmartHealthQuote (unless content is
                specifically made for redistribution)
              </li>
            </ul>
          </section>

          <section>
            <h2 className="text-xl font-semibold text-blue-800 mb-3 dark:text-blue-300">
              3. User Accounts
            </h2>
            <p className="text-gray-700 mb-3 dark:text-gray-300">
              When you create an account with us, you guarantee that the
              information you provide is accurate, complete, and current at all
              times. Inaccurate, incomplete, or obsolete information may result
              in the immediate termination of your account on our service.
            </p>
            <p className="text-gray-700 mb-3 dark:text-gray-300">
              You are responsible for maintaining the confidentiality of your
              account and password, including but not limited to the restriction
              of access to your computer and/or account. You agree to accept
              responsibility for any and all activities or actions that occur
              under your account and/or password.
            </p>
            <p className="text-gray-700 dark:text-gray-300">
              We reserve the right to refuse service, terminate accounts, remove
              or edit content, or cancel orders at our sole discretion.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-semibold text-blue-800 mb-3 dark:text-blue-300">
              4. Insurance Quotes
            </h2>
            <p className="text-gray-700 mb-3 dark:text-gray-300">
              The insurance quotes provided through our service are estimates
              based on the information you provide and our AI algorithms. These
              quotes are for informational purposes only and do not constitute
              an offer of insurance coverage.
            </p>
            <p className="text-gray-700 mb-3 dark:text-gray-300">
              Actual insurance policies, premiums, and coverage may vary based
              on additional underwriting criteria, medical examinations, or
              other factors determined by the insurance providers. We do not
              guarantee that you will be able to obtain insurance at the quoted
              rates.
            </p>
            <p className="text-gray-700 dark:text-gray-300">
              You acknowledge that you must work directly with insurance
              providers or licensed insurance agents to apply for and obtain
              actual insurance coverage.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-semibold text-blue-800 mb-3 dark:text-blue-300">
              5. Accuracy of Information
            </h2>
            <p className="text-gray-700 mb-3 dark:text-gray-300">
              You agree to provide accurate, current, and complete information
              during the registration process and with all interactions with our
              chat interface. The quality and accuracy of the insurance quotes
              we provide depend on the accuracy of the information you provide.
            </p>
            <p className="text-gray-700 dark:text-gray-300">
              Providing false or misleading information may result in inaccurate
              quotes and could potentially constitute insurance fraud, which may
              have legal consequences. We are not responsible for quotes that
              are inaccurate due to incorrect information provided by users.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-semibold text-blue-800 mb-3 dark:text-blue-300">
              6. Limitation of Liability
            </h2>
            <p className="text-gray-700 mb-3 dark:text-gray-300">
              In no event shall SmartHealthQuote, nor its directors,
              employees, partners, agents, suppliers, or affiliates, be liable
              for any indirect, incidental, special, consequential or punitive
              damages, including without limitation, loss of profits, data, use,
              goodwill, or other intangible losses, resulting from:
            </p>
            <ul className="list-disc pl-6 space-y-2 text-gray-700 dark:text-gray-300">
              <li>
                Your access to or use of or inability to access or use the
                service;
              </li>
              <li>Any conduct or content of any third party on the service;</li>
              <li>Any content obtained from the service; and</li>
              <li>
                Unauthorized access, use, or alteration of your transmissions or
                content, whether based on warranty, contract, tort (including
                negligence), or any other legal theory, whether or not we have
                been informed of the possibility of such damage.
              </li>
            </ul>
          </section>

          <section>
            <h2 className="text-xl font-semibold text-blue-800 mb-3 dark:text-blue-300">
              7. Disclaimer
            </h2>
            <p className="text-gray-700 mb-3 dark:text-gray-300">
              Your use of the service is at your sole risk. The service is
              provided on an "AS IS" and "AS AVAILABLE" basis. The service is
              provided without warranties of any kind, whether express or
              implied, including, but not limited to, implied warranties of
              merchantability, fitness for a particular purpose,
              non-infringement, or course of performance.
            </p>
            <p className="text-gray-700 dark:text-gray-300">
              SmartHealthQuote, its subsidiaries, affiliates, and its
              licensors do not warrant that:
            </p>
            <ul className="list-disc pl-6 space-y-2 text-gray-700 dark:text-gray-300">
              <li>
                The service will function uninterrupted, secure, or available at
                any particular time or location;
              </li>
              <li>Any errors or defects will be corrected;</li>
              <li>
                The service is free of viruses or other harmful components; or
              </li>
              <li>
                The results of using the service will meet your requirements.
              </li>
            </ul>
          </section>

          <section>
            <h2 className="text-xl font-semibold text-blue-800 mb-3 dark:text-blue-300">
              8. Governing Law
            </h2>
            <p className="text-gray-700 dark:text-gray-300">
              These Terms shall be governed and construed in accordance with the
              laws of the United States, without regard to its conflict of law
              provisions. Our failure to enforce any right or provision of these
              Terms will not be considered a waiver of those rights. If any
              provision of these Terms is held to be invalid or unenforceable by
              a court, the remaining provisions of these Terms will remain in
              effect. These Terms constitute the entire agreement between us
              regarding our Service, and supersede and replace any prior
              agreements we might have had between us regarding the Service.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-semibold text-blue-800 mb-3 dark:text-blue-300">
              9. Changes to Terms
            </h2>
            <p className="text-gray-700 mb-3 dark:text-gray-300">
              We reserve the right, at our sole discretion, to modify or replace
              these Terms at any time. If a revision is material, we will
              provide at least 30 days' notice prior to any new terms taking
              effect. What constitutes a material change will be determined at
              our sole discretion.
            </p>
            <p className="text-gray-700 dark:text-gray-300">
              By continuing to access or use our Service after any revisions
              become effective, you agree to be bound by the revised terms. If
              you do not agree to the new terms, you are no longer authorized to
              use the Service.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-semibold text-blue-800 mb-3 dark:text-blue-300">
              10. Contact Us
            </h2>
            <p className="text-gray-700 dark:text-gray-300">
              If you have any questions about this privacy policy or our privacy
              practices, please contact us at:
            </p>
            <div className="bg-blue-50 p-4 rounded-lg mt-3 dark:bg-blue-950">
              <p className="text-gray-700 dark:text-gray-300">
                <strong>Email:</strong> kmitisshit@gmail.com
              </p>
              <p className="text-gray-700 dark:text-gray-300">
                <strong>Postal Address:</strong> Keshav Memorial Institute of
                Technology, Narayanaguda
              </p>
              <p className="text-gray-700 dark:text-gray-300">
                <strong>Phone:</strong> 100
              </p>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
};

export default TermsOfService;
