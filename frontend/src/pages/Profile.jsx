import { useEffect, useState } from "react";
import "../styles/Profile.css";
import { getCurrentUser, updateProfile } from "../api";

export default function Profile() {
  const [user, setUser] = useState(null);
  const [editing, setEditing] = useState(false);

  const [form, setForm] = useState({
    username: "",
    password: "",
  });

  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState("");

  // fetch current user
  useEffect(() => {
    async function fetchUser() {
      try {
        const me = await getCurrentUser();
        setUser(me);

        setForm((prev) => ({
          ...prev,
          username: me?.username || "",
          password: "",
        }));
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    }
    fetchUser();
  }, []);

  function handleChange(e) {
    const { name, value } = e.target;
    setForm((prev) => ({ ...prev, [name]: value }));
  }

  async function handleSave(e) {
    e.preventDefault();
    setSaving(true);
    setMessage("");

    try {
      // build payload: only send fields user changed
      const payload = {};
      if (form.username?.trim() && form.username.trim() !== user?.username) {
        payload.username = form.username.trim();
      }
      if (form.password?.trim()) {
        payload.password = form.password.trim();
      }

      if (Object.keys(payload).length === 0) {
        setMessage("No changes to save.");
        setSaving(false);
        setEditing(false);
        return;
      }

      // call backend PUT /auth/me via api.js
      const updated = await updateProfile(payload);

      setUser(updated);

      // keep sidebar/Home greeting consistent
      if (updated?.username) {
        localStorage.setItem("username", updated.username);
      }

      setEditing(false);
      setForm((prev) => ({ ...prev, password: "" }));
      setMessage("Profile updated successfully.");
    } catch (err) {
      console.error(err);

      // try to show backend detail if available
      const detail =
        err?.response?.data?.detail ||
        err?.message ||
        "Failed to update profile.";

      setMessage(detail);
    } finally {
      setSaving(false);
    }
  }

  function handleCancel() {
    setEditing(false);
    setMessage("");

    // restore form to current user values
    setForm({
      username: user?.username || "",
      password: "",
    });
  }

  function getInitials() {
    const src = user?.username || user?.email || "User";
    return String(src).slice(0, 2).toUpperCase();
  }

  if (loading) {
    return (
      <div className="profile-page">
        <p>Loading profile...</p>
      </div>
    );
  }

  return (
    <div className="profile-page">
      {/* Top card */}
      <div className="profile-header-card">
        <div className="profile-avatar">{getInitials()}</div>

        <div className="profile-main-info">
          <h1>{user?.username || "User"}</h1>
          <p>{user?.email || "—"}</p>
          <span className="profile-badge">Logged in</span>
        </div>

        <div className="profile-actions">
          {!editing ? (
            <button
              className="profile-edit-btn"
              onClick={() => {
                setMessage("");
                setEditing(true);
              }}
            >
              <i className="bx bx-edit-alt"></i> Edit profile
            </button>
          ) : (
            <button className="profile-cancel-btn" onClick={handleCancel}>
              Cancel
            </button>
          )}
        </div>
      </div>

      <div className="profile-grid">
        {/* left: profile details */}
        <div className="profile-card">
          <h2>Profile details</h2>
          <p className="profile-hint">
            This information will be used across FetoVision.
          </p>

          {!editing ? (
            <div className="profile-details-view">
              <div className="profile-field">
                <label>Username</label>
                <p>{user?.username || "-"}</p>
              </div>

              <div className="profile-field">
                <label>Email</label>
                <p>{user?.email || "-"}</p>
              </div>

              <div className="profile-field">
                <label>Full name</label>
                <p>{user?.full_name || user?.username || "-"}</p>
              </div>

              <div className="profile-field">
                <label>Role</label>
                <p>{user?.role || "User"}</p>
              </div>
            </div>
          ) : (
            <form className="profile-form" onSubmit={handleSave}>
              <div className="input-group">
                <label htmlFor="username">Username</label>
                <input
                  id="username"
                  name="username"
                  value={form.username}
                  onChange={handleChange}
                  required
                />
              </div>

              {/* Email is display-only*/}
              <div className="input-group">
                <label>Email</label>
                <input value={user?.email || ""} disabled />
              </div>

              {/* Optional password change */}
              <div className="input-group">
                <label htmlFor="password">New password</label>
                <input
                  id="password"
                  name="password"
                  type="password"
                  value={form.password}
                  onChange={handleChange}
                  placeholder="Leave blank to keep current password"
                />
              </div>

              <button className="profile-save-btn" disabled={saving}>
                {saving ? "Saving..." : "Save changes"}
              </button>
            </form>
          )}

          {message ? <p className="profile-message">{message}</p> : null}
        </div>

        {/* right: account */}
        <div className="profile-card">
          <h2>Account</h2>
          <p className="profile-hint">
            Manage security and other account-related settings.
          </p>

          <div className="account-row">
            <div>
              <h4>Password</h4>
              <p>You can change password using “Edit profile”.</p>
            </div>
          </div>

          <div className="account-row">
            <div>
              <h4>Joined</h4>
              <p>{user?.created_at ? user.created_at : "Not available"}</p>
            </div>
          </div>

          <div className="account-row danger">
            <div>
              <h4>Logout</h4>
              <p>Sign out from this device.</p>
            </div>
            <button
              className="outline-danger"
              onClick={() => {
                localStorage.removeItem("token");
                localStorage.removeItem("username");
                window.location.href = "/login";
              }}
            >
              Logout
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
