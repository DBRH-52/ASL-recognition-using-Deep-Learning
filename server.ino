// server.ino

#include <Arduino.h>
#include <WiFi.h>
#include <ESPAsyncWebServer.h>
#include <TensorFlowLite_ESP32.h>

const char* ssid = "ssid";
const char* password = "password";

AsyncWebServer server(80);

#include "asl_model_data.h"

// Image dimensions and channels
constexpr int k_image_size = 64;  // Update to match a model input size
constexpr int k_channels = 1;   // Assuming grayscale images

// TFL (tensorflow lite) interpreter and input tensor
tflite::MicroErrorReporter micro_error_reporter;
const tflite::Model* model = tflite::GetModel(asl_model_data);
tflite::MicroInterpreter* interpreter = new tflite::MicroInterpreter(model, micro_error_reporter, k_image_size * k_image_size * k_channels, nullptr);
TfLiteTensor* input_tensor = interpreter->input(0);

void setup() {
  Serial.begin(115200);

  // Connect to Wi-Fi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi...");
  }

  // Set up web server
  server.on("/", HTTP_GET, [](AsyncWebServerRequest *request){
    request->send(SPIFFS, "/index.html", "text/html");
  });

  server.on("/upload", HTTP_POST, hanle_upload);

  server.begin();
}

void loop() {
  // 
}

void hanle_upload(AsyncWebServerRequest *request, String filename, size_t index, uint8_t *data, size_t len, bool final) {
  // Process image data and perform ASL recognition using TFL 
  // Update the TFL input tensor with the image data
  if (index == 0) {
    // Assuming len is always divisible by 4 (32-bit alignment)
    memcpy(input_tensor->data.f, data, len);
  }

  if (final) {
    // Run inference using the interpreter
    interpreter->Invoke();

    // Get recognition results
    int predicted_class = tflite::GetMaxResult(input_tensor->data.f, interpreter->outputs()[0]->dims[0]);

    // Convert numeric value back to letter
    char predicted_label = 'A' + predicted_class;

    // Send recognition results as an HTTP response
    String response = "Predicted letter: ";
    response += predicted_label;
    request->send(200, "text/plain", response);
  }
}
