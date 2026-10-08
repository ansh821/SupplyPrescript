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

const initialForm = {
  Benefit_per_order: 0,
  Sales_per_customer: 0,
  Customer_Zipcode: 0,
  Order_Id: 100000,
  Order_Item_Discount: 0,
  Order_Item_Discount_Rate: 0,
  Order_Item_Product_Price: 0,
  Order_Item_Profit_Ratio: 0,
  Order_Item_Quantity: 1,
  Sales: 0,
  Order_Item_Total: 0,
  Order_Profit_Per_Order: 0,
  Order_Zipcode: 0,
  Product_Price: 0,
  order_year: 2017,
  order_month: 1,
  order_day: 1,
  order_day_of_week: 0,
  order_week: 1,

  Type: "DEBIT",
  Category_Name: "Sporting Goods",
  Customer_City: "Caguas",
  Customer_Country: "Estados Unidos",
  Customer_Segment: "Consumer",
  Customer_State: "PR",
  Department_Name: "Fan Shop",
  Market: "USCA",
  Order_City: "Caguas",
  Order_Country: "Estados Unidos",
  Order_Region: "Caribbean",
  Order_State: "PR",
  Product_Name: "Field & Stream Sportsman 16 Gun Fire Safe",
  Shipping_Mode: "Standard Class",
};

const numericFields = [
  "Benefit_per_order",
  "Sales_per_customer",
  "Customer_Zipcode",
  "Order_Id",
  "Order_Item_Discount",
  "Order_Item_Discount_Rate",
  "Order_Item_Product_Price",
  "Order_Item_Profit_Ratio",
  "Order_Item_Quantity",
  "Sales",
  "Order_Item_Total",
  "Order_Profit_Per_Order",
  "Order_Zipcode",
  "Product_Price",
  "order_year",
  "order_month",
  "order_day",
  "order_day_of_week",
  "order_week",
];

const categoricalFields = [
  "Type",
  "Category_Name",
  "Customer_City",
  "Customer_Country",
  "Customer_Segment",
  "Customer_State",
  "Department_Name",
  "Market",
  "Order_City",
  "Order_Country",
  "Order_Region",
  "Order_State",
  "Product_Name",
  "Shipping_Mode",
];

function App() {
  const [form, setForm] = useState(initialForm);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleChange = (field, value) => {
    setForm((previous) => ({
      ...previous,
      [field]: value,
    }));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();

    setLoading(true);
    setError("");
    setResult(null);

    const payload = {};

    numericFields.forEach((field) => {
      payload[field] = Number(form[field]);
    });

    categoricalFields.forEach((field) => {
      payload[field] = form[field];
    });

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail
            ? Array.isArray(data.detail)
              ? data.detail.map((item) => item.msg).join(", ")
              : data.detail
            : "Prediction request failed."
        );
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

  const updateNumber = (field, value) => {
    handleChange(field, value);
  };

  const riskLevel = result?.prediction?.risk_level || "";
  const riskClass = riskLevel.toLowerCase();

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
              explainable AI, and operational recommendations to identify
              potentially delayed shipments before they become critical.
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

          <form className="prediction-panel" onSubmit={handleSubmit}>
            <div className="form-section">
              <div className="form-section-title">
                <h4>Order & Financial Information</h4>
                <span>Numeric model inputs</span>
              </div>

              <div className="form-grid">
                {numericFields.map((field) => (
                  <div className="form-field" key={field}>
                    <label htmlFor={field}>
                      {field.replaceAll("_", " ")}
                    </label>

                    <input
                      id={field}
                      type="number"
                      step="any"
                      value={form[field]}
                      onChange={(e) =>
                        updateNumber(field, e.target.value)
                      }
                    />
                  </div>
                ))}
              </div>
            </div>

            <div className="form-section">
              <div className="form-section-title">
                <h4>Order & Customer Information</h4>
                <span>Categorical model inputs</span>
              </div>

              <div className="form-grid">
                {categoricalFields.map((field) => (
                  <div className="form-field" key={field}>
                    <label htmlFor={field}>
                      {field.replaceAll("_", " ")}
                    </label>

                    <input
                      id={field}
                      type="text"
                      value={form[field]}
                      onChange={(e) =>
                        handleChange(field, e.target.value)
                      }
                    />
                  </div>
                ))}
              </div>
            </div>

            <div className="form-actions">
              <button type="submit" disabled={loading}>
                {loading ? (
                  <>
                    <RefreshCw className="spin" size={18} />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Activity size={18} />
                    Predict Delivery Risk
                  </>
                )}
              </button>
            </div>

            {error && (
              <div className="error-message">
                <AlertTriangle size={18} />
                <span>{error}</span>
              </div>
            )}

            {result && (
              <div className="result-area">
                <div className={`risk-card ${riskClass}`}>
                  <div className="risk-card-top">
                    <div>
                      <span className="result-label">
                        PREDICTED RISK
                      </span>

                      <h4>{riskLevel}</h4>
                    </div>

                    <div className="risk-icon">
                      {riskLevel === "HIGH" ? (
                        <AlertTriangle size={30} />
                      ) : riskLevel === "MEDIUM" ? (
                        <Clock3 size={30} />
                      ) : (
                        <CheckCircle size={30} />
                      )}
                    </div>
                  </div>

                  <div className="probability">
                    <span>Risk Probability</span>

                    <strong>
                      {(
                        result.prediction.risk_probability * 100
                      ).toFixed(2)}
                      %
                    </strong>
                  </div>

                  <div className="progress">
                    <div
                      style={{
                        width: `${Math.min(
                          result.prediction.risk_probability * 100,
                          100
                        )}%`,
                      }}
                    />
                  </div>
                </div>

                <div className="decision-card">
                  <span className="result-label">
                    RECOMMENDED DECISION
                  </span>

                  <div className="decision-row">
                    <div>
                      <span>Priority</span>
                      <strong>{result.decision.priority}</strong>
                    </div>

                    <div>
                      <span>Recommended Action</span>
                      <strong>
                        {result.decision.recommended_action}
                      </strong>
                    </div>
                  </div>
                </div>

                {result.explanation?.top_factors && (
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
          </form>
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
                model&apos;s decision.
              </p>
            </div>

            <div className="workflow-card">
              <div className="workflow-number">04</div>
              <CheckCircle size={22} />
              <h4>Prescribe</h4>
              <p>
                SupplyPrescript converts the prediction into a practical
                operational action.
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