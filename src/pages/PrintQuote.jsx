import { useEffect } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { FileText, Printer, ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";

const PrintQuote = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { quoteData, userDetails } = location.state || {};

  useEffect(() => {
    // If no state, go back
    if (!quoteData) {
      navigate("/");
    }
  }, [quoteData, navigate]);

  if (!quoteData) return null;

  return (
    <div className="min-h-dvh bg-background p-2 sm:p-4 md:p-8 font-sans">
      <div className="mx-auto max-w-3xl rounded-xl border border-border bg-card p-4 sm:p-6 md:p-10 shadow-sm print:m-0 print:max-w-none print:border-none print:shadow-none print:p-0 print:bg-white">
        
        {/* Action Bar (hidden when printing) */}
        <div className="mb-6 sm:mb-8 flex flex-col-reverse sm:flex-row sm:items-center justify-between gap-4 border-b border-border pb-4 sm:pb-6 print:hidden">
          <Button variant="ghost" onClick={() => navigate(-1)} className="gap-2 w-full sm:w-auto">
            <ArrowLeft className="h-4 w-4" /> Back
          </Button>
          <Button onClick={() => window.print()} className="gap-2 w-full sm:w-auto">
            <Printer className="h-4 w-4" /> Print Document
          </Button>
        </div>

        {/* Printable Content */}
        <div className="print-content text-text-primary">
          <div className="mb-8 sm:mb-10 text-center">
            <h1 className="text-h2 sm:text-display font-bold text-text-primary print:text-black">SmartHealthQuote</h1>
            <p className="mt-1 sm:mt-2 text-small sm:text-base text-text-secondary print:text-gray-600">Official Health Insurance Estimate</p>
          </div>

          <div className="mb-8 sm:mb-10 grid grid-cols-1 sm:grid-cols-2 gap-6 sm:gap-8 border-b border-border pb-6 sm:pb-8">
            <div>
              <h3 className="mb-3 sm:mb-4 text-h4 sm:text-h3 font-semibold text-text-primary print:text-black">Applicant Details</h3>
              <ul className="space-y-2 text-small text-text-secondary print:text-gray-800">
                <li className="flex justify-between sm:justify-start sm:gap-2"><span className="font-medium text-text-primary print:text-black">Age:</span> <span>{userDetails?.age || "N/A"} years</span></li>
                <li className="flex justify-between sm:justify-start sm:gap-2"><span className="font-medium text-text-primary print:text-black">Gender:</span> <span>{userDetails?.gender || "N/A"}</span></li>
                <li className="flex justify-between sm:justify-start sm:gap-2"><span className="font-medium text-text-primary print:text-black">Location:</span> <span>{userDetails?.location || "N/A"}</span></li>
                <li className="flex justify-between sm:justify-start sm:gap-2"><span className="font-medium text-text-primary print:text-black">Plan Type:</span> <span>{userDetails?.planType || "N/A"}</span></li>
              </ul>
            </div>
            <div>
              <h3 className="mb-3 sm:mb-4 text-h4 sm:text-h3 font-semibold text-text-primary print:text-black">Coverage Details</h3>
              <ul className="space-y-2 text-small text-text-secondary print:text-gray-800">
                <li className="flex justify-between sm:justify-start sm:gap-2"><span className="font-medium text-text-primary print:text-black">Sum Insured:</span> <span>{userDetails?.sumInsured || "N/A"}</span></li>
                <li className="flex justify-between sm:justify-start sm:gap-2"><span className="font-medium text-text-primary print:text-black">Members Covered:</span> <span>{userDetails?.numberOfInsuredMembers || 1}</span></li>
                <li className="flex justify-between sm:justify-start sm:gap-2"><span className="font-medium text-text-primary print:text-black">Policy Term:</span> <span>{userDetails?.policyTermYears || "1 Year"}</span></li>
              </ul>
            </div>
          </div>

          <div>
            <h2 className="mb-4 sm:mb-6 text-h3 sm:text-h2 font-bold text-text-primary print:text-black">Estimated Premiums</h2>
            
            <div className="overflow-x-auto rounded-lg border border-border print:border-gray-300">
              <table className="w-full text-left text-small min-w-[280px]">
                <thead className="bg-muted print:bg-gray-100">
                  <tr>
                    <th className="px-3 py-3 sm:px-6 sm:py-4 font-semibold text-text-primary print:text-black">Payment Frequency</th>
                    <th className="px-3 py-3 sm:px-6 sm:py-4 text-right font-semibold text-text-primary print:text-black">Premium Amount</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border print:divide-gray-300">
                  <tr>
                    <td className="px-3 py-3 sm:px-6 sm:py-4 text-text-secondary print:text-gray-800">Monthly</td>
                    <td className="px-3 py-3 sm:px-6 sm:py-4 text-right font-medium text-text-primary print:text-black">₹{quoteData?.monthlyINR?.toLocaleString("en-IN") || "0"}</td>
                  </tr>
                  <tr>
                    <td className="px-3 py-3 sm:px-6 sm:py-4 text-text-secondary print:text-gray-800">Quarterly</td>
                    <td className="px-3 py-3 sm:px-6 sm:py-4 text-right font-medium text-text-primary print:text-black">₹{quoteData?.quarterlyINR?.toLocaleString("en-IN") || "0"}</td>
                  </tr>
                  <tr className="bg-primary/5 print:bg-gray-50">
                    <td className="px-3 py-3 sm:px-6 sm:py-4 font-semibold text-primary print:text-black">Annual (Recommended)</td>
                    <td className="px-3 py-3 sm:px-6 sm:py-4 text-right text-base sm:text-h3 font-bold text-primary print:text-black">₹{quoteData?.yearlyINR?.toLocaleString("en-IN") || "0"}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
          
          <div className="mt-8 sm:mt-12 text-center text-caption text-text-tertiary print:text-gray-500">
            <p>This is an estimated quote based on the information provided and does not constitute a final offer of insurance.</p>
            <p className="mt-1">Generated on {new Date().toLocaleDateString()}</p>
          </div>
        </div>

      </div>
    </div>
  );
};

export default PrintQuote;
