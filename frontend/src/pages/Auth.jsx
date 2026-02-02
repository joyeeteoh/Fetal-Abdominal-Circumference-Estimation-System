import { useLocation, useNavigate, Link } from "react-router-dom";
import { useState, useEffect } from "react";
import { loginUser, registerUser } from "../api";
import "../styles/Auth.css";

export default function Auth() {
  const location = useLocation();
  const navigate = useNavigate();

  const isRegister = location.pathname === "/register";

  // login form state
  const [loginForm, setLoginForm] = useState({
    email: "",
    password: "",
  });
  const [loginLoading, setLoginLoading] = useState(false);
  const [loginErr, setLoginErr] = useState("");

  // register form state
  const [registerForm, setRegisterForm] = useState({
    username: "",
    email: "",
    password: "",
  });
  const [registerLoading, setRegisterLoading] = useState(false);
  const [registerErr, setRegisterErr] = useState("");

  // scroll to top when route changes
  useEffect(() => {
    window.scrollTo(0, 0);
  }, [location.pathname]);

  // ===== LOGIN SUBMIT =====
  async function handleLoginSubmit(e) {
    e.preventDefault();
    setLoginLoading(true);
    setLoginErr("");

    try {
      const data = await loginUser(loginForm);

      localStorage.setItem("token", data.access_token);

      if (data.username && data.username.trim() !== "") {
        localStorage.setItem("username", data.username);
      }

      navigate("/app");
    } catch (err) {
      setLoginErr(err.message);
    } finally {
      setLoginLoading(false);
    }
  }

  // ===== REGISTER SUBMIT =====
  async function handleRegisterSubmit(e) {
    e.preventDefault();
    setRegisterLoading(true);
    setRegisterErr("");

    try {
      await registerUser(registerForm);

      localStorage.setItem("username", registerForm.username);

      navigate("/login");
    } catch (err) {
      setRegisterErr(err.message);
    } finally {
      setRegisterLoading(false);
    }
  }

  return (
    <div className="auth-screen">
      <div className={`auth-container ${isRegister ? "right-panel-active" : ""}`}>
        {/* SIGN IN FORM */}
        <div className="form-container sign-in-container">
          <form onSubmit={handleLoginSubmit} className="auth-form-flex">
            <h2 className="auth-title">Sign In</h2>
            <div className="social-container">
              <button type="button" className="social-btn">G</button>
              <button type="button" className="social-btn">f</button>
              <button type="button" className="social-btn">in</button>
            </div>
            <p className="small-text">or use your email password</p>

            <input
              type="email"
              placeholder="Email"
              value={loginForm.email}
              onChange={(e) => setLoginForm({ ...loginForm, email: e.target.value })}
              required
            />
            <input
              type="password"
              placeholder="Password"
              value={loginForm.password}
              onChange={(e) => setLoginForm({ ...loginForm, password: e.target.value })}
              required
            />

            <span className="forgot-link">Forget Your Password?</span>

            {loginErr && <p className="error-msg">{loginErr}</p>}

            <button type="submit" className="purple-btn" disabled={loginLoading}>
              {loginLoading ? "Signing in..." : "SIGN IN"}
            </button>
          </form>
        </div>

        {/* SIGN UP FORM */}
        <div className="form-container sign-up-container">
          <form onSubmit={handleRegisterSubmit} className="auth-form-flex">
            <h2 className="auth-title">Create Account</h2>
            <div className="social-container">
              <button type="button" className="social-btn">G</button>
              <button type="button" className="social-btn">f</button>
              <button type="button" className="social-btn">in</button>
            </div>
            <p className="small-text">or use your email for registration</p>

            <input
              type="text"
              placeholder="Username"
              value={registerForm.username}
              onChange={(e) =>
                setRegisterForm({ ...registerForm, username: e.target.value })
              }
              required
            />
            <input
              type="email"
              placeholder="Email"
              value={registerForm.email}
              onChange={(e) =>
                setRegisterForm({ ...registerForm, email: e.target.value })
              }
              required
            />
            <input
              type="password"
              placeholder="Password"
              value={registerForm.password}
              onChange={(e) =>
                setRegisterForm({ ...registerForm, password: e.target.value })
              }
              minLength={8}
              required
            />

            {registerErr && <p className="error-msg">{registerErr}</p>}

            <button type="submit" className="purple-btn" disabled={registerLoading}>
              {registerLoading ? "Creating account..." : "SIGN UP"}
            </button>
          </form>
        </div>

        {/* TOPIC OVERLAY */}
        <div className="overlay-container">
          <div className="overlay">
            <div className="overlay-panel overlay-right">
              <h2>Welcome!</h2>
              <p>
                Sign up to start predicting abdominal
                circumference from ultrasound images.
              </p>
              <Link to="/register" className="transparent-btn">
                SIGN UP
              </Link>
            </div>

            <div className="overlay-panel overlay-left">
              <h2>Welcome Back!</h2>
              <p>Log in to access AI-driven abdominal circumference prediction.</p>
              <Link to="/login" className="transparent-btn">
                SIGN IN
              </Link>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
