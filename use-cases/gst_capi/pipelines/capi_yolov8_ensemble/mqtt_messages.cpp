//*****************************************************************************
// Copyright 2024 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

extern "C" {
    #include "MQTTClient.h"
}

#include "mqtt_messages.hpp"

class MQTTPublisher {
    public:
        MQTTPublisher(const string& clientID, const string& mqttHostname, int mqttPort) {
            // construct broker address:
            stringstream brokerAddr;
            brokerAddr << "tcp://" << mqttHostname << ":" << mqttPort;
            this->brokerAddr = brokerAddr.str();

            // Initialize MQTT client:
            MQTTClient_connectOptions conn_opts = MQTTClient_connectOptions_initializer;
            conn_opts.keepAliveInterval = 30;
            conn_opts.cleansession = 1;
            MQTTClient_create(&client, this->brokerAddr.c_str(), clientID.c_str(), MQTTCLIENT_PERSISTENCE_NONE, NULL);
            
            // try to connect to MQTT broker:
            int statusCode;
            if ((statusCode = MQTTClient_connect(client, &conn_opts)) != MQTTCLIENT_SUCCESS) {
                cerr << "failed to connect to the MQTT broker: " << this->brokerAddr << " status code = " << statusCode << endl;
                exit(EXIT_FAILURE);
            }

            cout << "Connected to MQTT broker: " << this->brokerAddr << endl;
        }

        ~MQTTPublisher() {
            // disconnect and clean up:
            MQTTClient_disconnect(client, CONNECTION_TIME_OUT);
            MQTTClient_destroy(&client);

            cout << "Disconnected from MQTT broker: " << brokerAddr << endl;
        }

        void publish(const string& topic, const string& mqttMessages) {
            // prepare messages to publish:
            MQTTClient_message msgPub = MQTTClient_message_initializer;
            msgPub.payload = (void*)mqttMessages.c_str();
            msgPub.payloadlen = (int)mqttMessages.length();
            msgPub.qos = 1;
            msgPub.retained = 0;
            
            MQTTClient_deliveryToken token;
            // publish:
            MQTTClient_publishMessage(client, topic.c_str(), &msgPub, &token);
            cout << "publishing messages to MQTT broker: " << brokerAddr << endl;
            cout << "topic: " << topic.c_str() << " messages: " << mqttMessages << endl;
            cout << "waiting for up to " << (int)(CONNECTION_TIME_OUT/1000) << " seconds for publishing..." << endl;
            // check delivery:
            int statusCode = MQTTClient_waitForCompletion(client, token, CONNECTION_TIME_OUT);
            cout << "delivered token: " << token << " status code = " << statusCode << endl;
        }

    private:
        MQTTClient client;
        string brokerAddr;
};
