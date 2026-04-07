import React from "react";
import { Button } from "@/components/ui/button";
import { ArrowLeft } from "lucide-react";
import { useEffect } from "react";
import { NavLink } from "react-router";

const PrivacyPolicy = () => {
  useEffect(() => {
    document.title = "Privacy Policy";
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
          Privacy Policy
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
              Welcome to SmartHealthQuote. We respect your privacy and are
              committed to protecting your personal data. This privacy policy
              will inform you about how we look after your personal data when
              you visit our website and tell you about your privacy rights and
              how the law protects you.
            </p>
            <p className="text-gray-700 dark:text-gray-300">
              This privacy policy aims to give you information on how
              SmartHealthQuote collects and processes your personal data
              through your use of this website, including any data you may
              provide through this website when you sign up for our service, use
              our chat interface, or request an insurance quote.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-semibold text-blue-800 mb-3 dark:text-blue-300">
              2. Data We Collect
            </h2>
            <p className="text-gray-700 mb-3 dark:text-gray-300">
              We may collect, use, store and transfer different kinds of
              personal data about you which we have grouped together as follows:
            </p>
            <ul className="list-disc pl-6 space-y-2 text-gray-700 dark:text-gray-300">
              <li>
                <strong>Identity Data</strong> includes first name, last name,
                username or similar identifier, date of birth, and gender.
              </li>
              <li>
                <strong>Contact Data</strong> includes email address, telephone
                numbers, and physical address.
              </li>
              <li>
                <strong>Health Data</strong> includes information about your
                health history, pre-existing conditions, and lifestyle factors
                that are relevant to insurance quotes.
              </li>
              <li>
                <strong>Technical Data</strong> includes internet protocol (IP)
                address, your login data, browser type and version, time zone
                setting and location, browser plug-in types and versions,
                operating system and platform, and other technology on the
                devices you use to access this website.
              </li>
              <li>
                <strong>Usage Data</strong> includes information about how you
                use our website and services.
              </li>
            </ul>
          </section>

          <section>
            <h2 className="text-xl font-semibold text-blue-800 mb-3 dark:text-blue-300">
              3. How We Use Your Data
            </h2>
            <p className="text-gray-700 mb-3 dark:text-gray-300">
              We will only use your personal data when the law allows us to.
              Most commonly, we will use your personal data in the following
              circumstances:
            </p>
            <ul className="list-disc pl-6 space-y-2 text-gray-700 dark:text-gray-300">
              <li>
                To provide you with personalized insurance quotes based on your
                inputs.
              </li>
              <li>
                To improve our AI chat system and make it more responsive to
                user needs.
              </li>
              <li>
                To communicate with you about our services and provide customer
                support.
              </li>
              <li>
                To comply with legal obligations and protect our legal rights.
              </li>
              <li>
                To share relevant information with insurance providers when you
                explicitly request to be connected with them.
              </li>
            </ul>
          </section>

          <section>
            <h2 className="text-xl font-semibold text-blue-800 mb-3 dark:text-blue-300">
              4. Data Security
            </h2>
            <p className="text-gray-700 mb-3 dark:text-gray-300">
              We have put in place appropriate security measures to prevent your
              personal data from being accidentally lost, used, or accessed in
              an unauthorized way, altered, or disclosed. In addition, we limit
              access to your personal data to those employees, agents,
              contractors, and other third parties who have a business need to
              know.
            </p>
            <p className="text-gray-700 dark:text-gray-300">
              We have put in place procedures to deal with any suspected
              personal data breach and will notify you and any applicable
              regulator of a breach where we are legally required to do so.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-semibold text-blue-800 mb-3 dark:text-blue-300">
              5. Data Retention
            </h2>
            <p className="text-gray-700 dark:text-gray-300">
              We will only retain your personal data for as long as necessary to
              fulfill the purposes we collected it for, including for the
              purposes of satisfying any legal, accounting, or reporting
              requirements. To determine the appropriate retention period for
              personal data, we consider the amount, nature, and sensitivity of
              the personal data, the potential risk of harm from unauthorized
              use or disclosure of your personal data, the purposes for which we
              process your personal data, and whether we can achieve those
              purposes through other means, and the applicable legal
              requirements.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-semibold text-blue-800 mb-3 dark:text-blue-300">
              6. Your Legal Rights
            </h2>
            <p className="text-gray-700 mb-3 dark:text-gray-300">
              Under certain circumstances, you have rights under data protection
              laws in relation to your personal data, including the right to:
            </p>
            <ul className="list-disc pl-6 space-y-2 text-gray-700 dark:text-gray-300">
              <li>Request access to your personal data.</li>
              <li>Request correction of your personal data.</li>
              <li>Request erasure of your personal data.</li>
              <li>Object to processing of your personal data.</li>
              <li>Request restriction of processing your personal data.</li>
              <li>Request transfer of your personal data.</li>
              <li>Right to withdraw consent.</li>
            </ul>
            <p className="text-gray-700 mt-3 dark:text-gray-300">
              If you wish to exercise any of these rights, please contact us
              using the details provided below.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-semibold text-blue-800 mb-3 dark:text-blue-300">
              7. Cookies
            </h2>
            <p className="text-gray-700 mb-3 dark:text-gray-300">
              Our website uses cookies to distinguish you from other users of
              our website. This helps us to provide you with a good experience
              when you browse our website and also allows us to improve our
              site. A cookie is a small file of letters and numbers that we
              store on your browser or the hard drive of your computer if you
              agree.
            </p>
            <p className="text-gray-700 dark:text-gray-300">
              You can set your browser to refuse all or some browser cookies, or
              to alert you when websites set or access cookies. If you disable
              or refuse cookies, please note that some parts of this website may
              become inaccessible or not function properly.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-semibold text-blue-800 mb-3 dark:text-blue-300">
              8. Contact Us
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

          <section>
            <h2 className="text-xl font-semibold text-blue-800 mb-3 dark:text-blue-300">
              9. Changes to This Policy
            </h2>
            <p className="text-gray-700 dark:text-gray-300">
              We may update this privacy policy from time to time. We will
              notify you of any changes by posting the new privacy policy on
              this page and updating the "Last Updated" date at the top of this
              policy. You are advised to review this privacy policy periodically
              for any changes.
            </p>
          </section>
        </div>
      </div>
    </div>
  );
};

export default PrivacyPolicy;
