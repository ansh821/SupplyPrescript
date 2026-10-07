import { useState } from "react";
import {
  Activity,
  AlertTriangle,
  CheckCircle,
  Clock3,
  Package,
  RefreshCw,
  ShieldAlert,
  Sparkles,
  TrendingUp,
} from "lucide-react";
import "./App.css";

const API_BASE_URL = "http://127.0.0.1:8000";

function App() {
  const [orderIndex, setOrderIndex] = useState("2");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const checkRisk = async () => {
    if (!orderIndex.trim()) {
      setError("Please enter an order index.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(
        `${API_BASE_URL}/predict/${orderIndex.trim()}`
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Prediction request failed.");
      }

      setResult(data);
    } catch (err) {
      setError(
        err.message ||
          "Unable to connect to the SupplyPrescript backend."
      );
    } finally {
      setLoading(false);
    }
  };

  const riskClass = result?.risk_level
    ? result.risk_level.toLowerCase()
    : "";

  return (
    <div className="app">
      <header className="navbar">
        <div className="brand">
          <div className="brand-icon">
            <Activity size={22} />
          </div>

          <div>
            <h1>SupplyPrescript</h1>
            <span>AI Supply Chain Risk Intelligence</span>
          </div>
        </div>

        <div className="system-status">
          <span className="status-dot"></span>
          API System Online
        </div>
      </header>

      <main className="dashboard">
        <section className="hero">
          <div>
            <div className="eyebrow">
              <Sparkles size={15} />
              Predictive Supply Intelligence
            </div>

            <h2>
              Predict delivery risk.
              <br />
              <span>Prescribe the next action.</span>
            </h2>

            <p>
              SupplyPrescript combines machine learning prediction,
              explainable AI, and operational recommendations to help
              identify potentially delayed shipments before they become
              critical.
            </p>
          </div>

          <div className="hero-card">
            <TrendingUp size={30} />
            <strong>XGBoost + SHAP</strong>
            <span>Prediction & Explainability Engine</span>
          </div>
        </section>

        <section className="stats-grid">
          <div className="stat-card">
            <div className="stat-icon blue">
              <Package size={20} />
            </div>
            <div>
              <span>Prediction Model</span>
              <strong>XGBoost</strong>
            </div>
          </div>

          <div className="stat-card">
            <div className="stat-icon purple">
              <ShieldAlert size={20} />
            </div>
            <div>
              <span>Risk Intelligence</span>
              <strong>3 Risk Levels</strong>
            </div>
          </div>

          <div className="stat-card">
            <div className="stat-icon green">
              <Sparkles size={20} />
            </div>
            <div>
              <span>Explainability</span>
              <strong>SHAP Enabled</strong>
            </div>
          </div>

          <div className="stat-card">
            <div className="stat-icon orange">
              <Clock3 size={20} />
            </div>
            <div>
              <span>Decision Engine</span>
              <strong>Real-time</strong>
            </div>
          </div>
        </section>

        <section className="prediction-section">
          <div className="section-heading">
            <div>
              <span className="section-label">RISK PREDICTION</span>
              <h3>Analyze an order</h3>
            </div>

            <span className="model-badge">XGBoost Model</span>
          </div>

          <div className="prediction-panel">
            <div className="input-area">
              <label htmlFor="orderIndex">Order Index</label>

              <div className="input-row">
                <input
                  id="orderIndex"
                  type="number"
                  min="0"
                  value={orderIndex}
                  onChange={(e) => setOrderIndex(e.target.value)}
                  placeholder="Enter order index"
                />

                <button
                  onClick={checkRisk}
                  disabled={loading}
                >
                  {loading ? (
                    <>
                      <RefreshCw className="spin" size={18} />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Activity size={18} />
                      Predict Risk
                    </>
                  )}
                </button>
              </div>

              <p className="input-help">
                Enter an index from the grouped test dataset.
              </p>

              {error && (
                <div className="error-message">
                  <AlertTriangle size={18} />
                  {error}
                </div>
              )}
            </div>

            {result && (
              <div className="result-area">
                <div className={`risk-card ${riskClass}`}>
                  <div className="risk-card-top">
                    <div>
                      <span className="result-label">
                        PREDICTED RISK
                      </span>

                      <h4>{result.risk_level}</h4>
                    </div>

                    <div className="risk-icon">
                      {result.risk_level === "HIGH" ? (
                        <AlertTriangle size={30} />
                      ) : result.risk_level === "MEDIUM" ? (
                        <Clock3 size={30} />
                      ) : (
                        <CheckCircle size={30} />
                      )}
                    </div>
                  </div>

                  <div className="probability">
                    <span>Risk Probability</span>

                    <strong>
                      {(result.risk_probability * 100).toFixed(2)}%
                    </strong>
                  </div>

                  <div className="progress">
                    <div
                      style={{
                        width: `${Math.min(
                          result.risk_probability * 100,
                          100
                        )}%`,
                      }}
                    />
                  </div>
                </div>

                <div className="decision-card">
                  <span className="result-label">RECOMMENDED DECISION</span>

                  <div className="decision-row">
                    <div>
                      <span>Priority</span>
                      <strong>{result.priority}</strong>
                    </div>

                    <div>
                      <span>Recommended Action</span>
                      <strong>{result.recommended_action}</strong>
                    </div>
                  </div>
                </div>

                {result.explanation &&
                  result.explanation.top_factors && (
                    <div className="explanation-card">
                      <div className="explanation-heading">
                        <div>
                          <span className="result-label">
                            EXPLAINABLE AI
                          </span>
                          <h4>Why did the model predict this?</h4>
                        </div>

                        <Sparkles size={22} />
                      </div>

                      <div className="factor-list">
                        {result.explanation.top_factors.map(
                          (factor, index) => (
                            <div
                              className="factor"
                              key={`${factor.feature}-${index}`}
                            >
                              <div className="factor-number">
                                {index + 1}
                              </div>

                              <div className="factor-content">
                                <strong>{factor.feature}</strong>

                                <span
                                  className={
                                    factor.direction === "increases"
                                      ? "increase"
                                      : "decrease"
                                  }
                                >
                                  {factor.direction === "increases"
                                    ? "Increases risk"
                                    : "Reduces risk"}
                                </span>
                              </div>

                              <div className="contribution">
                                {factor.contribution > 0 ? "+" : ""}
                                {Number(
                                  factor.contribution
                                ).toFixed(4)}
                              </div>
                            </div>
                          )
                        )}
                      </div>
                    </div>
                  )}
              </div>
            )}
          </div>
        </section>

        <section className="workflow">
          <div className="section-heading">
            <div>
              <span className="section-label">DECISION WORKFLOW</span>
              <h3>From prediction to prescription</h3>
            </div>
          </div>

          <div className="workflow-grid">
            <div className="workflow-card">
              <div className="workflow-number">01</div>
              <Activity size={22} />
              <h4>Predict</h4>
              <p>
                XGBoost analyzes order-level features and estimates
                late-delivery probability.
              </p>
            </div>

            <div className="workflow-card">
              <div className="workflow-number">02</div>
              <ShieldAlert size={22} />
              <h4>Classify</h4>
              <p>
                The prediction is converted into LOW, MEDIUM, or HIGH
                operational risk.
              </p>
            </div>

            <div className="workflow-card">
              <div className="workflow-number">03</div>
              <Sparkles size={22} />
              <h4>Explain</h4>
              <p>
                SHAP identifies the strongest factors influencing the
                model's decision.
              </p>
            </div>

            <div className="workflow-card">
              <div className="workflow-number">04</div>
              <CheckCircle size={22} />
              <h4>Prescribe</h4>
              <p>
                SupplyPrescript converts the prediction into a
                practical operational action.
              </p>
            </div>
          </div>
        </section>
      </main>

      <footer>
        <span>SupplyPrescript</span>
        <span>AI-powered supply chain decision intelligence</span>
      </footer>
    </div>
  );
}

export default App;