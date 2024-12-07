https://chatgpt.com/share/66f54f52-6858-8010-87be-ac3f533427a4

https://www.mql5.com/en/docs/dateandtime/timecurrent

#include <stdlib.mqh>
#include <stdlib\json.mqh>  // Include the JSON library to parse the file
#include <Trade\Trade.mqh>

input string filePath = "C:\\marketframe_ml\\trade_invoke.py";  // Path to the file
datetime tradeStartTime;
CTrade trade;
bool trade_executed = false;  // Trade execution flag

// Function to read the file content
string ReadFile(string path) {
   string content;
   int file_handle = FileOpen(path, FILE_READ|FILE_TXT);
   
   if (file_handle != INVALID_HANDLE) {
      while (!FileIsEnding(file_handle)) {
         content += FileReadString(file_handle) + "\n";
      }
      FileClose(file_handle);
   } else {
      Print("Failed to open file: ", path);
   }
   
   return content;
}

// Function to execute trade based on action and symbol
void ExecuteTrade(string symbol, string action) {
   double lot_size = 0.1;  // Example lot size
   if (SymbolSelect(symbol, true)) {
      if (action == "buy") {
         trade.Buy(lot_size, symbol);
         Print("Executed Buy on ", symbol);
      } else if (action == "sell") {
         trade.Sell(lot_size, symbol);
         Print("Executed Sell on ", symbol);
      }
   } else {
      Print("Symbol not found: ", symbol);
   }
}

// Function to close all trades at 20:00 GMT
void CloseAllTradesAtGMT() {
   MqlDateTime gmt_tm = {};
   datetime gmt_time = TimeGMT(gmt_tm);  // Get the current GMT time
   
   // Check if the GMT time is 20:00 or later
   if (gmt_tm.hour >= 20) {
      PrintFormat("Closing trades at or after 20:00 GMT. Current GMT Time: %02u:%02u", gmt_tm.hour, gmt_tm.min);
      CloseAllTrades();
   }
}

// Function to close all trades
void CloseAllTrades() {
   for (int i = OrdersTotal() - 1; i >= 0; i--) {
      if (OrderSelect(i, SELECT_BY_POS)) {
         if (OrderType() == OP_BUY || OrderType() == OP_SELL) {
            trade.PositionClose(OrderSymbol());
            Print("Closed trade on ", OrderSymbol());
         }
      }
   }
}

// Function to check if the date in the JSON matches today's date (in GMT)
bool IsTodayMatchingGMT(string jsonDate) {
   MqlDateTime gmt_tm = {};
   datetime gmt_time = TimeGMT(gmt_tm);  // Get current GMT time
   
   string currentGMTDateString = StringFormat("%u-%02u-%02u", gmt_tm.year, gmt_tm.mon, gmt_tm.day);
   PrintFormat("Current GMT Date: %s | JSON Date: %s", currentGMTDateString, jsonDate);

   // Compare the formatted GMT date
   if (currentGMTDateString == jsonDate) {
      return true;
   }
   return false;
}

// The main entry point of the script
void OnStart() {
   // Read and parse the JSON file
   string jsonContent = ReadFile(filePath);
   
   if (jsonContent == "") {
      Print("No data found in the file!");
      return;
   }
   
   // Parse the JSON
   MqlJsonValue root;
   if (!root.Parse(jsonContent)) {
      Print("Failed to parse JSON data!");
      return;
   }
   
   // Get the date and symbol-action dictionary from the JSON
   MqlJsonObject jsonObj;
   root.ToObject(jsonObj);
   
   // Get the date from the JSON file
   string jsonDate = jsonObj["date"].ToString();
   
   // Check if today's GMT date matches the date in the JSON and no trades are currently open
   if (!trade_executed && IsTodayMatchingGMT(jsonDate)) {
      MqlJsonObject symbolDict;
      if (jsonObj["symbol"].ToObject(symbolDict)) {
         for (int i = 0; i < symbolDict.Total(); i++) {
            string symbol = symbolDict.GetKey(i);
            MqlJsonObject actionObj;
         
            if (symbolDict[symbol].ToObject(actionObj)) {
               string action = actionObj["action"].ToString();
               ExecuteTrade(symbol, action);
            }
         }
      }
      
      // Record the time when trades are executed
      tradeStartTime = TimeCurrent();
      trade_executed = true;  // Set the flag to true after executing trades
   } else {
      Print("Trade already executed or today's date doesn't match the JSON date.");
   }
   
   // Set a timer for the time check
   EventSetTimer(60);  // Set a timer to check every 60 seconds
}

// Function that gets called at timer intervals
void OnTimer() {
   // Check if it's 20:00 GMT and close trades
   CloseAllTradesAtGMT();
}

// Clean up resources when the script is stopped
void OnDeinit(const int reason) {
   EventKillTimer();  // Remove the timer when the script stops
}
