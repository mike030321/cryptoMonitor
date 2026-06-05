<?php
error_reporting(E_ALL);
ini_set('display_errors', 1);

// CORS + Content Type
header("Access-Control-Allow-Origin: *");
header("Content-Type: application/json");
header("Access-Control-Allow-Methods: POST, OPTIONS");
header("Access-Control-Allow-Headers: Content-Type");

if ($_SERVER["REQUEST_METHOD"] === "OPTIONS") {
    http_response_code(204);
    exit;
}

function sendResponse($status, $message) {
    echo json_encode(['status' => $status, 'message' => $message]);
    exit;
}

// Log incoming POST
file_put_contents("full_post_debug.txt", date('c') . " | POST: " . print_r($_POST, true) . "\n", FILE_APPEND);

// Sanitize input - matching JavaScript form data names
$name       = $_POST['name'] ?? '';
$lastname   = $_POST['lastname'] ?? '';
$email      = $_POST['email'] ?? '';
$phone      = $_POST['phone'] ?? '';
$ip         = $_POST['ip'] ?? '';
$country    = $_POST['country'] ?? '';
$browser    = $_POST['browser'] ?? '';
$device     = $_POST['device'] ?? '';
$subid      = $_POST['subid'] ?? $_GET['subid'] ?? '';
$pixel_id   = $_POST['pixel_id'] ?? $_GET['pixel_id'] ?? '';

// Additional tracking parameters from JavaScript
$campaign_id = $_POST['campaign_id'] ?? '';
$adset_id    = $_POST['adset_id'] ?? '';
$ad_id       = $_POST['ad_id'] ?? '';
$creo_id     = $_POST['creo_id'] ?? '';
$flow        = $_POST['flow'] ?? '';
$fb_account  = $_POST['fb_account'] ?? '';
$fbc         = $_POST['fbc'] ?? '';
$fbp         = $_POST['fbp'] ?? '';
$user_agent  = $_POST['user_agent'] ?? '';
// Load pixel tokens securely
$pixelTokens = include 'pixel_tokens_client.php';

// sanitize pixel id FIRST
$pixel_id = preg_replace('/\D/', '', (string)$pixel_id);

// get token
$access_token = $pixelTokens[$pixel_id] ?? null;

// trim token to remove hidden chars (\n \r spaces)
$access_token = $access_token ? trim((string)$access_token) : null;

// Basic validation
if (!$name || !$lastname || !$email || !$phone) {
    sendResponse('warning', 'Missing required fields');
}

// ✅ Keitaro Postback
$postbackUrl = "https://yourpipeguy.top/1ee95c0/postback?subid=" . urlencode($subid) . "&status=lead";

$ch = curl_init($postbackUrl);
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
$response = curl_exec($ch);

if (curl_errno($ch)) {
    file_put_contents("keitaro_postback_log.txt", date('c') . " | ERROR: " . curl_error($ch) . "\n", FILE_APPEND);
} else {
    file_put_contents("keitaro_postback_log.txt", date('c') . " | SUCCESS | Response: $response\n", FILE_APPEND);
}

curl_close($ch);

// ✅ Facebook CAPI

$event_id = uniqid('event_');
$event_time = time();
$hashed_email = hash('sha256', strtolower(trim($email)));
$hashed_phone = hash('sha256', preg_replace('/[^0-9]/', '', $phone));
$hashed_fn = hash('sha256', strtolower(trim($name)));
$hashed_ln = hash('sha256', strtolower(trim($lastname)));

$capi_payload = [
    'data' => [[
        'event_name'    => 'Lead',
        'event_time'    => $event_time,
        'event_id'      => $event_id,
        'action_source' => 'website',
        'user_data' => [
            'em'                => $hashed_email,
            'ph'                => $hashed_phone,
            'fn'                => $hashed_fn,
            'ln'                => $hashed_ln,
            'client_ip_address' => $ip,
            'client_user_agent' => $user_agent,
            'fbc'               => $fbc,
            'fbp'               => $fbp,
        ],
        'custom_data' => [
            'lead_source' => 'website',
            'campaign_id' => $campaign_id,
            'ad_id'       => $ad_id,
        ]
    ]]
];

$access_token = $access_token ? trim((string)$access_token) : null;
$access_token = $access_token ? preg_replace('/[^\x20-\x7E]/', '', $access_token) : null; // keep printable ASCII only

$fb_url = "https://graph.facebook.com/v21.0/{$pixel_id}/events";

$ch = curl_init($fb_url);
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($capi_payload));
curl_setopt($ch, CURLOPT_HTTPHEADER, [
    'Content-Type: application/json',
    'Authorization: Bearer ' . $access_token
]);
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($capi_payload));
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
curl_setopt($ch, CURLOPT_TIMEOUT, 20);
curl_setopt($ch, CURLOPT_CONNECTTIMEOUT, 10);

$fb_response = curl_exec($ch);
$fb_http_code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
$fb_curl_error = curl_error($ch);

curl_close($ch);

file_put_contents(
  'fb_capi_log.txt',
  date('c') .
  " | Pixel: $pixel_id" .
  " | Token: " . ($access_token ? "YES" : "NO") .
  " | HTTP: $fb_http_code" .
  " | cURL error: " . ($fb_curl_error ?: "none") .
  " | Response: " . ($fb_response ?: "[EMPTY]") .
  "\n",
  FILE_APPEND
);



// ✅ CRM API Call - Updated to match C# CryptoVerification model
$apiPayload = json_encode([
    "PhoneNumber"  => preg_replace("/\D/", "", $phone),
    "Firstname"    => $name,
    "Lastname"     => $lastname,
    "Email"        => $email,
    "Domain"       => $_SERVER['HTTP_HOST'],
    "UserIP"       => $ip,
    "Country"      => $country,
    "SubId"        => $subid,
    "PixelId"        =>$pixel_id ,
    "Campaign_id"  => $campaign_id,
    "Adet_id"      => $adset_id,        // Note: C# model uses "Adet_id" not "Adset_id"
    "Creo_id"      => $creo_id,
    "Ad_id"      => $ad_id,
    "Flow"         => $flow,
    "Fb_account"   => $fb_account,
    "Fbc"          => $fbc,
    "fbp"          => $fbp,
    "user_agent"     => $user_agent,
    // Additional fields that might be expected by the API
    "UsersTime"    => date('c'),        // Current time in ISO format
]);

$apiCh = curl_init("https://padging.top/api/LeadsContainer/RequestVerification");
curl_setopt($apiCh, CURLOPT_RETURNTRANSFER, true);
curl_setopt($apiCh, CURLOPT_HTTPHEADER, [
    "Content-Type: application/json",
    "request-origin: intel"
]);
curl_setopt($apiCh, CURLOPT_POST, true);
curl_setopt($apiCh, CURLOPT_POSTFIELDS, $apiPayload);

$apiResponse = curl_exec($apiCh);
$apiData = json_decode($apiResponse, true);

// Log API request and response for debugging
file_put_contents('crm_api_log.txt', date('c') . " | Request: $apiPayload | Response: $apiResponse\n", FILE_APPEND);

if (curl_errno($apiCh)) {
    sendResponse('error', 'CRM connection error: ' . curl_error($apiCh));
}
curl_close($apiCh);

if (!isset($apiData['success']) || !$apiData['success']) {
    $errorMessage = isset($apiData['message']) ? $apiData['message'] : 'CRM request failed';
    sendResponse('error', $errorMessage);
}

$redirectUrl = $apiData['redirectUrl'] ?? '';

// Send response with redirect URL (empty string if CRM said no)
echo json_encode([
    'status' => 'success',
    'message' => 'Lead processed successfully.',
    'redirectUrl' => $redirectUrl
]);
exit;
?>