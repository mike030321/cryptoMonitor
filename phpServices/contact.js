const params = new URLSearchParams(window.location.search);
if (params.has("subid")) {
  document.cookie = `subid=${params.get("subid")}; path=/;`;
}
if (params.has("pixel")) {
  document.cookie = `pixel=${params.get("pixel")}; path=/;`;
}

function getCookie(name) {
  const match = document.cookie.match("(^|;)\\s*" + name + "\\s*=\\s*([^;]+)");
  return match ? decodeURIComponent(match[2]) : null;
}

// ✅ ADD THESE DETECTION FUNCTIONS
function getBrowser() {
  const userAgent = navigator.userAgent;

  if (userAgent.includes("Chrome") && !userAgent.includes("Edg")) {
    return "Chrome";
  } else if (userAgent.includes("Firefox")) {
    return "Firefox";
  } else if (userAgent.includes("Safari") && !userAgent.includes("Chrome")) {
    return "Safari";
  } else if (userAgent.includes("Edg")) {
    return "Edge";
  } else if (userAgent.includes("Opera") || userAgent.includes("OPR")) {
    return "Opera";
  } else {
    return "Unknown";
  }
}

function getDevice() {
  const userAgent = navigator.userAgent;

  // Check if it's a tablet first
  if (
    /iPad/i.test(userAgent) ||
    (/Android/i.test(userAgent) && !/Mobile/i.test(userAgent))
  ) {
    return "Tablet";
  }

  // Check if it's a phone
  if (/Android|iPhone|iPod|BlackBerry|IEMobile|Opera Mini/i.test(userAgent)) {
    return "Phone";
  }

  // Default to desktop
  return "Desktop";
}

// =====================
// CURRENCY CONFIGURATION
// =====================
const CURRENCY_MAP = {
  US: { symbol: '$', locale: 'en-US' },
  CA: { symbol: 'CA$', locale: 'en-CA' },
  GB: { symbol: '£', locale: 'en-GB' },
  AU: { symbol: 'A$', locale: 'en-AU' },
  CH: { symbol: 'CHF ', locale: 'de-CH' },
  DK: { symbol: 'kr ', locale: 'da-DK' },
  SE: { symbol: 'kr ', locale: 'sv-SE' },
  NO: { symbol: 'kr ', locale: 'nb-NO' },
  PL: { symbol: 'zł ', locale: 'pl-PL' },
  // All others default to EUR
};
const DEFAULT_CURRENCY = { symbol: '€', locale: 'en-US' };
window.userCurrency = DEFAULT_CURRENCY;

function getCurrencyByCountry(countryCode) {
  return CURRENCY_MAP[countryCode?.toUpperCase()] || DEFAULT_CURRENCY;
}

// Add this validation function after the getDevice() function
function validatePhone(iti) {
  if (!iti) return false;

  const input = iti.a; // Get the input element
  const phoneNumber = input.value.trim();

  // Check if phone number is empty
  if (!phoneNumber) {
    return { valid: false, error: "Phone number is required" };
  }

  // Check if the number is valid using intl-tel-input validation
  if (!iti.isValidNumber()) {
    const errorCode = iti.getValidationError();
    let errorMessage = "Invalid phone number";

    // Map error codes to user-friendly messages
    switch (errorCode) {
      case intlTelInputUtils.validationError.TOO_SHORT:
        errorMessage = "Phone number is too short";
        break;
      case intlTelInputUtils.validationError.TOO_LONG:
        errorMessage = "Phone number is too long";
        break;
      case intlTelInputUtils.validationError.INVALID_COUNTRY_CODE:
        errorMessage = "Invalid country code";
        break;
      case intlTelInputUtils.validationError.NOT_A_NUMBER:
        errorMessage = "Please enter numbers only";
        break;
      default:
        errorMessage = "Please enter a valid phone number";
    }

    return { valid: false, error: errorMessage };
  }

  return { valid: true };
}

// Add visual feedback function
function showPhoneError(input, message) {
  // Remove any existing error
  const existingError = input.parentElement.querySelector(".phone-error");
  if (existingError) {
    existingError.remove();
  }

  // Add error class to input
  input.classList.add("phone-error-input");

  // Create and insert error message with absolute positioning
  const errorDiv = document.createElement("div");
  errorDiv.className = "phone-error";
  errorDiv.textContent = message;
  input.parentElement.style.position = "relative"; // Make parent relative
  input.parentElement.appendChild(errorDiv);
}

function clearPhoneError(input) {
  const existingError = input.parentElement.querySelector(".phone-error");
  if (existingError) {
    existingError.remove();
  }
  input.classList.remove("phone-error-input");
}

// Name validation function
function validateName(input, fieldName) {
  const value = input.value.trim();

  if (!value) {
    return { valid: false, error: `${fieldName} is required` };
  }

  if (value.length < 2) {
    return { valid: false, error: `${fieldName} must be at least 2 characters` };
  }

  if (/\d/.test(value)) {
    return { valid: false, error: `${fieldName} cannot contain numbers` };
  }

  if (!/^[a-zA-ZÀ-ÿ\s'-]+$/.test(value)) {
    return { valid: false, error: `${fieldName} can only contain letters` };
  }

  return { valid: true };
}

function showNameError(input, message) {
  const existingError = input.parentElement.querySelector(".name-error");
  if (existingError) {
    existingError.remove();
  }

  input.classList.add("name-error-input");

  const errorDiv = document.createElement("div");
  errorDiv.className = "name-error";
  errorDiv.textContent = message;
  input.parentElement.style.position = "relative";
  input.parentElement.appendChild(errorDiv);
}

function clearNameError(input) {
  const existingError = input.parentElement.querySelector(".name-error");
  if (existingError) {
    existingError.remove();
  }
  input.classList.remove("name-error-input");
}

// Enhanced fetch with timeout and better error handling
async function fetchWithTimeout(url, options = {}, timeout = 10000) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    return response;
  } catch (error) {
    clearTimeout(timeoutId);
    if (error.name === "AbortError") {
      throw new Error("Request timed out");
    }
    throw error;
  }
}

// Enhanced IP/Country detection with better error handling
let cachedIpData = null;

async function getIpAndCountry() {
  if (cachedIpData) return cachedIpData;

  let userIp = "";
  let userCountry = "Unknown";

  try {
    console.log("Getting IP info from IPinfo...");
    const ipInfoResponse = await fetchWithTimeout(
      "https://ipinfo.io/json?token=a91cf61e10058a",
      {},
      5000 // 5 second timeout
    );

    if (ipInfoResponse.ok) {
      const ipInfo = await ipInfoResponse.json();
      userIp = ipInfo.ip || "";
      userCountry = ipInfo.country || "Unknown";
      window.userCurrency = getCurrencyByCountry(ipInfo.country);
      console.log("IPinfo success:", { ip: userIp, country: userCountry });
      return { userIp, userCountry };
    }
    throw new Error("IPinfo returned non-200 status");
  } catch (error) {
    console.log("IPinfo failed, trying fallback:", error.message);

    try {
      const fallbackResponse = await fetchWithTimeout(
        "https://ipapi.co/json/",
        {},
        5000 // 5 second timeout
      );

      if (fallbackResponse.ok) {
        const fallbackData = await fallbackResponse.json();
        userCountry = fallbackData.country_name || "Unknown";
        userIp = fallbackData.ip || "";
        window.userCurrency = getCurrencyByCountry(fallbackData.country_code);
        console.log("Fallback success:", { ip: userIp, country: userCountry });
        return { userIp, userCountry };
      }
      throw new Error("Fallback service also failed");
    } catch (fallbackError) {
      console.log("Both IP services failed:", fallbackError.message);
      return { userIp: "", userCountry: "Unknown" };
    }
  }
}

document.addEventListener("DOMContentLoaded", function () {
  const phoneInputs = [document.querySelector("#phone1")];
  const itiInstances = [];

  phoneInputs.forEach((input) => {
    if (input) {
      const iti = window.intlTelInput(input, {
        initialCountry: "auto",
        separateDialCode: true,
        autoPlaceholder: "off",
        geoIpLookup: (success) => {
          // Using IPinfo with token
          fetchWithTimeout(
            "https://ipinfo.io/json?token=a91cf61e10058a",
            {},
            5000
          )
            .then((res) => {
              if (res.ok) {
                return res.json();
              }
              throw new Error("IPinfo failed");
            })
            .then((data) => success(data.country))
            .catch(() => success("us"));
        },
        utilsScript:
          "https://cdnjs.cloudflare.com/ajax/libs/intl-tel-input/17.0.8/js/utils.js",
      });
      itiInstances.push(iti);
    }
  });

  // Real-time validation for name fields
  document.querySelectorAll(".contact-form").forEach((form) => {
    const firstNameInput = form.querySelector("[name='name']");
    const lastNameInput = form.querySelector("[name='lastname']");

    if (firstNameInput) {
      firstNameInput.addEventListener("blur", function () {
        const validation = validateName(this, "First name");
        if (!validation.valid) {
          showNameError(this, validation.error);
        } else {
          clearNameError(this);
        }
      });

      firstNameInput.addEventListener("input", function () {
        clearNameError(this);
      });
    }

    if (lastNameInput) {
      lastNameInput.addEventListener("blur", function () {
        const validation = validateName(this, "Last name");
        if (!validation.valid) {
          showNameError(this, validation.error);
        } else {
          clearNameError(this);
        }
      });

      lastNameInput.addEventListener("input", function () {
        clearNameError(this);
      });
    }
  });

  phoneInputs.forEach((input, index) => {
    if (input) {
      input.addEventListener("blur", function () {
        const iti = itiInstances[index];
        const validation = validatePhone(iti);

        if (!validation.valid) {
          showPhoneError(input, validation.error);
        } else {
          clearPhoneError(input);
        }
      });

      // Clear error on input
      input.addEventListener("input", function () {
        clearPhoneError(input);
      });
    }
  });

  document.querySelectorAll(".contact-form").forEach((form, i) => {
    form.addEventListener("submit", async function (e) {
      e.preventDefault();

      const input = phoneInputs[i];
      const iti = itiInstances[i];

      // Get name inputs
      const firstNameInput = form.querySelector("[name='name']");
      const lastNameInput = form.querySelector("[name='lastname']");

      // Clear any existing errors
      clearPhoneError(input);
      clearNameError(firstNameInput);
      clearNameError(lastNameInput);

      // Validate first name
      const firstNameValidation = validateName(firstNameInput, "First name");
      if (!firstNameValidation.valid) {
        showNameError(firstNameInput, firstNameValidation.error);
        return;
      }

      // Validate last name
      const lastNameValidation = validateName(lastNameInput, "Last name");
      if (!lastNameValidation.valid) {
        showNameError(lastNameInput, lastNameValidation.error);
        return;
      }

      // Validate phone before proceeding
      const phoneValidation = validatePhone(iti);
      if (!phoneValidation.valid) {
        showPhoneError(input, phoneValidation.error);
        return;
      }

      // Get overlay elements
      const loadingOverlay = document.getElementById('formLoadingOverlay');
      const loadingText = document.getElementById('loadingText');
      const loadingSubtext = document.getElementById('loadingSubtext');

      // Disable submit button
      const submitButton = form.querySelector(
        'button[type="submit"], input[type="submit"]'
      );
      if (submitButton) {
        submitButton.disabled = true;
        submitButton.textContent = "Processing...";
      }

      // Show loading overlay
      if (loadingOverlay) {
        loadingOverlay.classList.add('active');
      }

      try {
        // Get IP and country
        const { userIp, userCountry } = await getIpAndCountry();

        const name = firstNameInput.value.trim();
        const lastname = lastNameInput.value.trim();
        const email = form.querySelector("[name='email']").value;
        const phone = iti.getNumber();
        const subid = params.get("subid") || "";
        const pixel = params.get("pixel") || "";

        const browser = getBrowser();
        const device = getDevice();
        const userAgent = navigator.userAgent;

        const campaignId = params.get("campaign_id") || getCookie("campaign_id") || "";
        const adsetId = params.get("adset_id") || getCookie("adset_id") || "";
        const adId = params.get("ad_id") || getCookie("ad_id") || "";
        const creoId = params.get("creo_id") || getCookie("creo_id") || "";
        const flow = params.get("flow") || getCookie("flow") || "";
        const fbAccount = params.get("fb_account") || getCookie("fb_account") || "";
        const fbc = params.get("fbc") || getCookie("fbc") || getCookie("_fbc") || "";
        const fbp = params.get("fbp") || getCookie("fbp") || getCookie("_fbp") || "";

        const formData = new FormData();
        formData.append("name", name);
        formData.append("lastname", lastname);
        formData.append("email", email);
        formData.append("phone", phone);
        formData.append("ip", userIp);
        formData.append("country", userCountry);
        formData.append("subid", subid);
        formData.append("pixel_id", pixel);
        formData.append("browser", browser);
        formData.append("device", device);
        formData.append("campaign_id", campaignId);
        formData.append("adset_id", adsetId);
        formData.append("ad_id", adId);
        formData.append("creo_id", creoId);
        formData.append("flow", flow);
        formData.append("fb_account", fbAccount);
        formData.append("fbc", fbc);
        formData.append("fbp", fbp);
        formData.append("user_agent", userAgent);

        console.log("Submitting form to mail.php...");

        // Submit form
        const response = await fetchWithTimeout(
          "mail.php",
          {
            method: "POST",
            body: formData,
          },
          30000
        );

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const result = await response.json();
        console.log("Form submission result:", result);

        // Update overlay to show redirecting message
        if (loadingText) {
          loadingText.textContent = t('redirecting') || 'Redirecting';
        }
        if (loadingSubtext) {
          loadingSubtext.textContent = t('almost_done') || 'Almost done! Taking you to the next step...';
        }

        const queryParams = new URLSearchParams({
          subid: subid,
          pixel: pixel,
          campaign_id: campaignId,
          adset_id: adsetId,
          ad_id: adId,
          creo_id: creoId,
          flow: flow,
          fb_account: fbAccount,
          fbc: fbc,
          fbp: fbp,
        }).toString();

        // ✅ Decide redirect URL based on CRM response
        let redirectUrl = '';

        // Only use CRM URL if status is success AND redirectUrl exists
        if (result.status === "success" && result.redirectUrl && result.redirectUrl.trim() !== "") {
          // CRM returned a valid URL - use it
          redirectUrl = result.redirectUrl + (result.redirectUrl.includes("?") ? "&" : "?") + queryParams;
        } else {
          // Any other case (no URL, error, rejected) - go to confirmation page
          redirectUrl = `thankyou.php?lang=${window.currentLang}&${queryParams}`;
        }

        // Small delay to show the redirecting message
        setTimeout(() => {
          window.location.href = redirectUrl;
        }, 800);
      } catch (fetchError) {
        console.error("Form submission error:", fetchError);

        // Update overlay for redirect
        if (loadingText) {
          loadingText.textContent = t('redirecting') || 'Redirecting';
        }
        if (loadingSubtext) {
          loadingSubtext.textContent = t('almost_done') || 'Almost done! Taking you to the next step...';
        }

        const queryParams = new URLSearchParams({
          subid: subid,
          pixel: pixel,
          campaign_id: campaignId,
          adset_id: adsetId,
          ad_id: adId,
          creo_id: creoId,
          flow: flow,
          fb_account: fbAccount,
          fbc: fbc,
          fbp: fbp,
        }).toString();

        // Network error - redirect to confirmation page
        setTimeout(() => {
          window.location.href = `thankyou.php?lang=${window.currentLang}&${queryParams}`;
        }, 800);
      } finally {
        // Re-enable submit button (only if we haven't redirected)
        if (submitButton && !loadingOverlay.classList.contains('active')) {
          submitButton.disabled = false;
          submitButton.textContent = t('form_submit') || "Submit";
        }
      }
    });
  });

  // Detect currency on page load immediately
  getIpAndCountry().then(() => {
    // Currency is now set, re-render metrics if they're already visible
    const profitDisplay = document.getElementById("profit_display");
    const totalProfitsEl = document.getElementById("total_profits_count");
    // These will update on next interval tick automatically
    // But force an immediate re-render of the profit calculator display
    const cur = window.userCurrency || { symbol: '€', locale: 'en-US' };
    if (profitDisplay && profitDisplay.textContent) {
      // Extract the number and reformat with correct currency
      const rawText = profitDisplay.textContent.replace(/[^0-9]/g, '');
      if (rawText) {
        profitDisplay.textContent = cur.symbol + parseInt(rawText).toLocaleString(cur.locale);
      }
    }
    if (totalProfitsEl && totalProfitsEl.textContent) {
      const rawText = totalProfitsEl.textContent.replace(/[^0-9]/g, '');
      if (rawText) {
        totalProfitsEl.textContent = cur.symbol + parseInt(rawText).toLocaleString(cur.locale);
      }
    }
    const avgProfitEl = document.getElementById("avg_profit_count");
    if (avgProfitEl && avgProfitEl.textContent) {
      const rawText = avgProfitEl.textContent.replace(/[^0-9]/g, '');
      if (rawText) {
        avgProfitEl.textContent = cur.symbol + parseInt(rawText).toLocaleString(cur.locale);
      }
    }
  });
});
