#include <Arduino.h>#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <Adafruit_Sensor.h>
#include <DHT.h>
#include <Wire.h>
#include <BH1750.h>
#include <Adafruit_INA226.h>

// -------------------------
// WiFi & API Config
// -------------------------
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";
const char* api_url = "http://YOUR_PC_IP:5000/predict";

// -------------------------
// Sensor Pins & Setup
// -------------------------
#define DHTPIN 4
#define DHTTYPE DHT22
DHT dht(DHTPIN, DHTTYPE);

BH1750 lightMeter;
Adafruit_INA226 inaSolar;
Adafruit_INA226 inaBattery;

// -------------------------
// Actuators
// -------------------------
#define FAN_PIN 15
#define LED_ALERT_PIN 2
#define BUZZER_PIN 16

// -------------------------
// Timing
// -------------------------
unsigned long lastSend = 0;
const unsigned long interval = 5000; // 5 sec

void setup() {
  Serial.begin(115200);
  WiFi.begin(ssid, password);

  // Wait for connection
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\n✅ WiFi connected!");

  // Sensors
  dht.begin();
  Wire.begin();
  lightMeter.begin();
  inaSolar.begin();
  inaBattery.begin();

  // Actuators
  pinMode(FAN_PIN, OUTPUT);
  pinMode(LED_ALERT_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
}

void loop() {
  if (millis() - lastSend > interval) {
    lastSend = millis();

    // -------------------------
    // 1️⃣ Read sensors
    // -------------------------
    float temp = dht.readTemperature();
    float hum = dht.readHumidity();
    float lux = lightMeter.readLightLevel();
    
    float vSolar = inaSolar.readBusVoltage();
    float iSolar = inaSolar.readCurrent();
    float pSolar = inaSolar.readPower();

    float vBatt = inaBattery.readBusVoltage();
    float iBatt = inaBattery.readCurrent();
    
    // -------------------------
    // 2️⃣ Send data to API
    // -------------------------
    if (WiFi.status() == WL_CONNECTED) {
      HTTPClient http;
      http.begin(api_url);
      http.addHeader("Content-Type", "application/json");

      // Prepare JSON payload
      StaticJsonDocument<256> doc;
      doc["temperature"] = temp;
      doc["humidity"] = hum;
      doc["lux"] = lux;
      doc["v_solar"] = vSolar;
      doc["i_solar"] = iSolar;
      doc["p_solar"] = pSolar;
      doc["v_batt"] = vBatt;
      doc["i_batt"] = iBatt;

      String payload;
      serializeJson(doc, payload);

      int httpResponseCode = http.POST(payload);
      if (httpResponseCode > 0) {
        String response = http.getString();
        Serial.println("✅ API Response: " + response);

        
        // -------------------------
        // 3️⃣ Parse predictions
        // -------------------------
        StaticJsonDocument<256> resDoc;
        deserializeJson(resDoc, response);
        float efficiency = resDoc["efficiency"];
        float power_pred = resDoc["power"];
        float duty_cycle = resDoc["duty_cycle"];
        float batt_opt = resDoc["battery_voltage_pred"];

        // -------------------------
        // 4️⃣ Control Logic
        // -------------------------
        // Efficiency alert
        if (efficiency < 0.85) {
          digitalWrite(LED_ALERT_PIN, HIGH);
          digitalWrite(BUZZER_PIN, HIGH);
        } else {
          digitalWrite(LED_ALERT_PIN, LOW);
          digitalWrite(BUZZER_PIN, LOW);
        }

        // PWM control for fan/light (scale 0-1 to 0-255)
        analogWrite(FAN_PIN, (int)(duty_cycle * 255));

        // Optional: battery optimization logic
        if (batt_opt < 12.0) {
          // shed non-critical loads
          analogWrite(FAN_PIN, 0);
          Serial.println("⚠️ Battery low, shedding loads!");
        }

      } else {
        Serial.println("❌ Error sending POST: " + String(httpResponseCode));
      }

      http.end();
    }
  }
}
