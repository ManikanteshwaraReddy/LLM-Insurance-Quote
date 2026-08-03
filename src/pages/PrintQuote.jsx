import { useEffect } from "react";
import { useNavigate, NavLink } from "react-router-dom";
import { FileText } from "lucide-react";
import { EmptyState } from "@/components/ui/empty-state";
import { Button } from "@/components/ui/button";

const PrintQuote = () => {
  const navigate = useNavigate();

  useEffect(() => {
    window.print();
    navigate("/");
  }, [navigate]);

  return (
    <div className="flex min-h-dvh items-center justify-center bg-background px-4 py-12">
      <EmptyState
        icon={FileText}
        title="Print Quote"
        description="This page is under development. Printing will be available soon."
        action={
          <NavLink to="/">
            <Button variant="outline">Back to Home</Button>
          </NavLink>
        }
      />
    </div>
  );
};

export default PrintQuote;
