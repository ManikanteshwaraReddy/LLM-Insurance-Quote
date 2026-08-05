import { useState } from "react";
import { NavLink, useNavigate } from "react-router-dom";
import {
  ChevronRight,
  Edit3,
  FileText,
  Lock,
  LogOut,
  User,
} from "lucide-react";
import { useAuth } from "@/lib/AuthContext";
import { changePassword, updateProfile } from "@/lib/api";
import { Alert } from "@/components/ui/alert";
import { Avatar } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import { EmptyState } from "@/components/ui/empty-state";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { PasswordInput } from "@/components/ui/password-input";
import {
  PasswordChecklist,
  PASSWORD_REQUIREMENTS,
} from "@/components/ui/password-checklist";
import { Select } from "@/components/ui/select";

function formatDate(dateStr) {
  if (!dateStr) return "—";
  return new Date(dateStr).toLocaleDateString("en-IN", {
    day: "numeric",
    month: "short",
    year: "numeric",
  });
}

const GENDER_OPTIONS = ["Male", "Female", "Non-binary", "Prefer not to say", "Other"];

function InfoRow({ label, value }) {
  return (
    <div className="py-2 border-b border-border/50 last:border-0">
      <p className="text-caption text-text-tertiary">{label}</p>
      <p className="truncate text-small font-medium text-text-primary">{value || "—"}</p>
    </div>
  );
}

const ProfilePage = () => {
  const { user, logout, updateUser } = useAuth();
  const navigate = useNavigate();

  const [activeTab, setActiveTab] = useState("profile");
  const [isEditingProfile, setIsEditingProfile] = useState(false);
  const [profileSaving, setProfileSaving] = useState(false);
  const [profileMessage, setProfileMessage] = useState(null);
  const [profileError, setProfileError] = useState(null);

  const [profileForm, setProfileForm] = useState({
    fullName: user?.fullName || "",
    phone: user?.phone || "",
    dob: user?.dob || "",
    gender: user?.gender || "",
    address: user?.address || "",
    city: user?.city || "",
    state: user?.state || "",
    pincode: user?.pincode || "",
  });

  const [pwSaving, setPwSaving] = useState(false);
  const [pwMessage, setPwMessage] = useState(null);
  const [pwError, setPwError] = useState(null);
  const [newPw, setNewPw] = useState("");
  const [showChecklist, setShowChecklist] = useState(false);

  if (!user) {
    return (
      <div className="container mx-auto max-w-5xl px-4 py-16 text-center">
        <EmptyState
          icon={User}
          title="Authentication Required"
          description="Sign in to access your profile and health quotes."
          action={
            <NavLink to="/auth">
              <Button size="lg">Sign In</Button>
            </NavLink>
          }
        />
      </div>
    );
  }

  const handleProfileSubmit = async (e) => {
    e.preventDefault();
    setProfileSaving(true);
    setProfileError(null);
    setProfileMessage(null);

    try {
      const updated = await updateProfile(profileForm);
      updateUser(updated);
      setProfileMessage("Profile updated successfully.");
      setIsEditingProfile(false);
    } catch (err) {
      setProfileError(err.message || "Failed to update profile.");
    } finally {
      setProfileSaving(false);
    }
  };

  const handlePasswordSubmit = async (e) => {
    e.preventDefault();
    setPwSaving(true);
    setPwError(null);
    setPwMessage(null);

    const fd = new FormData(e.target);
    const currentPassword = fd.get("currentPassword");
    const newPassword = fd.get("newPassword");
    const confirmPassword = fd.get("confirmPassword");

    if (newPassword !== confirmPassword) {
      setPwError("New passwords do not match.");
      setPwSaving(false);
      return;
    }

    if (PASSWORD_REQUIREMENTS.some((r) => !r.test(newPassword))) {
      setPwError("New password does not meet requirements.");
      setPwSaving(false);
      return;
    }

    try {
      await changePassword({ currentPassword, newPassword });
      setPwMessage("Password changed successfully.");
      e.target.reset();
      setNewPw("");
      setShowChecklist(false);
    } catch (err) {
      setPwError(err.message || "Failed to change password.");
    } finally {
      setPwSaving(false);
    }
  };

  const handleLogout = async () => {
    await logout();
    navigate("/");
  };

  return (
    <div className="min-h-dvh bg-background py-10">
      <div className="container mx-auto max-w-4xl px-4 sm:px-6 lg:px-8">
        {/* Header Profile Summary */}
        <div className="mb-8 rounded-xl border border-border bg-card p-6 flex flex-col sm:flex-row sm:items-center justify-between gap-6">
          <div className="flex items-center gap-4">
            <Avatar name={user.fullName} className="h-14 w-14 text-h3 font-bold" />
            <div>
              <h1 className="text-h2 font-bold text-text-primary">{user.fullName}</h1>
              <p className="text-small text-text-secondary">{user.email}</p>
              <p className="text-caption text-text-tertiary">Member since {formatDate(user.createdAt)}</p>
            </div>
          </div>

          <Button variant="outline" size="sm" onClick={handleLogout} className="gap-2 w-full sm:w-auto">
            <LogOut className="h-4 w-4" /> Sign Out
          </Button>
        </div>

        {/* Tab Navigation */}
        <div className="mb-8 border-b border-border flex gap-6">
          {[
            { id: "profile", label: "Profile & Personal Details" },
            { id: "quotes", label: "Saved Estimates" },
            { id: "security", label: "Security & Password" },
          ].map((tab) => (
            <button
              key={tab.id}
              type="button"
              onClick={() => setActiveTab(tab.id)}
              className={`pb-3 text-small font-medium transition-colors border-b-2 -mb-px ${
                activeTab === tab.id
                  ? "border-primary text-primary font-semibold"
                  : "border-transparent text-text-secondary hover:text-text-primary"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Tab 1: Profile & Account */}
        {activeTab === "profile" && (
          <div className="space-y-6">
            {profileMessage && (
              <Alert type="success" message={profileMessage} onClose={() => setProfileMessage(null)} />
            )}
            {profileError && (
              <Alert type="error" message={profileError} onClose={() => setProfileError(null)} />
            )}

            <div className="rounded-xl border border-border bg-card p-6">
              <div className="flex items-center justify-between border-b border-border pb-4 mb-6">
                <h2 className="text-h3 font-semibold text-text-primary">Personal Details</h2>
                {!isEditingProfile && (
                  <Button variant="ghost" size="sm" onClick={() => setIsEditingProfile(true)} className="gap-1.5">
                    <Edit3 className="h-3.5 w-3.5" /> Edit
                  </Button>
                )}
              </div>

              {!isEditingProfile ? (
                <div className="grid gap-x-8 gap-y-2 sm:grid-cols-2">
                  <InfoRow label="Full Name" value={user.fullName} />
                  <InfoRow label="Email Address" value={user.email} />
                  <InfoRow label="Phone Number" value={user.phone} />
                  <InfoRow label="Gender" value={user.gender} />
                  <InfoRow label="Date of Birth" value={formatDate(user.dob)} />
                  <InfoRow label="Location" value={user.city ? `${user.city}, ${user.state}` : ""} />
                </div>
              ) : (
                <form onSubmit={handleProfileSubmit} className="space-y-4">
                  <div className="grid gap-4 sm:grid-cols-2">
                    <div>
                      <Label htmlFor="fullName" required>Full Name</Label>
                      <Input
                        id="fullName"
                        value={profileForm.fullName}
                        onChange={(e) => setProfileForm({ ...profileForm, fullName: e.target.value })}
                        required
                      />
                    </div>
                    <div>
                      <Label htmlFor="phone">Phone Number</Label>
                      <Input
                        id="phone"
                        type="tel"
                        value={profileForm.phone}
                        onChange={(e) => setProfileForm({ ...profileForm, phone: e.target.value })}
                      />
                    </div>
                    <div>
                      <Label htmlFor="dob">Date of Birth</Label>
                      <Input
                        id="dob"
                        type="date"
                        value={profileForm.dob}
                        onChange={(e) => setProfileForm({ ...profileForm, dob: e.target.value })}
                      />
                    </div>
                    <div>
                      <Label htmlFor="gender">Gender</Label>
                      <Select
                        id="gender"
                        options={GENDER_OPTIONS}
                        value={profileForm.gender}
                        onChange={(e) => setProfileForm({ ...profileForm, gender: e.target.value })}
                      />
                    </div>
                  </div>

                  <div className="flex flex-col-reverse sm:flex-row sm:items-center justify-end gap-3 pt-4 border-t border-border">
                    <Button variant="outline" type="button" onClick={() => setIsEditingProfile(false)} className="w-full sm:w-auto">
                      Cancel
                    </Button>
                    <Button type="submit" disabled={profileSaving} className="w-full sm:w-auto">
                      {profileSaving ? "Saving..." : "Save Changes"}
                    </Button>
                  </div>
                </form>
              )}
            </div>
          </div>
        )}

        {/* Tab 2: Saved Quotes */}
        {activeTab === "quotes" && (
          <div className="rounded-xl border border-border bg-card p-6">
            <EmptyState
              icon={FileText}
              title="No Saved Estimates"
              description="Complete the guided health assessment to generate and save personalized quotes."
              action={
                <NavLink to="/chat">
                  <Button className="gap-2">
                    <span>Start Quote Assessment</span>
                    <ChevronRight className="h-4 w-4" />
                  </Button>
                </NavLink>
              }
            />
          </div>
        )}

        {/* Tab 3: Security & Password */}
        {activeTab === "security" && (
          <div className="space-y-6">
            {pwMessage && <Alert type="success" message={pwMessage} onClose={() => setPwMessage(null)} />}
            {pwError && <Alert type="error" message={pwError} onClose={() => setPwError(null)} />}

            <div className="rounded-xl border border-border bg-card p-6">
              <h2 className="text-h3 font-semibold text-text-primary mb-6 border-b border-border pb-4">
                Change Password
              </h2>
              <form onSubmit={handlePasswordSubmit} className="max-w-md space-y-4">
                <div>
                  <Label htmlFor="currentPassword" required>Current Password</Label>
                  <PasswordInput id="currentPassword" name="currentPassword" required />
                </div>

                <div>
                  <Label htmlFor="newPassword" required>New Password</Label>
                  <PasswordInput
                    id="newPassword"
                    name="newPassword"
                    value={newPw}
                    onChange={(e) => {
                      setNewPw(e.target.value);
                      setShowChecklist(true);
                    }}
                    required
                  />
                  {showChecklist && <PasswordChecklist password={newPw} />}
                </div>

                <div>
                  <Label htmlFor="confirmPassword" required>Confirm New Password</Label>
                  <PasswordInput id="confirmPassword" name="confirmPassword" required />
                </div>

                <Button type="submit" disabled={pwSaving} className="w-full sm:w-auto mt-2">
                  {pwSaving ? "Updating Password..." : "Update Password"}
                </Button>
              </form>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ProfilePage;
