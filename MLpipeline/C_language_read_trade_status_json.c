#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <jansson.h>  // Include the Jansson library for JSON parsing

// Function to get the current date in YYYY-MM-DD format
void get_current_date(char *date_str) {
    time_t t = time(NULL);
    struct tm tm = *localtime(&t);
    sprintf(date_str, "%04d-%02d-%02d", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday);
}

int main() {
    // Open the JSON file
    FILE *file = fopen("trade_status.json", "r");
    if (!file) {
        perror("Unable to open file");
        return 1;
    }

    // Read the entire JSON file into a buffer
    fseek(file, 0, SEEK_END);
    long len = ftell(file);
    fseek(file, 0, SEEK_SET);
    char *data = (char *)malloc(len + 1);
    fread(data, 1, len, file);
    fclose(file);
    data[len] = '\0';  // Null-terminate the JSON data

    // Parse the JSON data
    json_t *root;
    json_error_t error;
    root = json_loads(data, 0, &error);
    free(data);
    if (!root) {
        fprintf(stderr, "Error parsing JSON: %s\n", error.text);
        return 1;
    }

    // Get the current date
    char current_date[11];  // YYYY-MM-DD format
    get_current_date(current_date);

    // Extract the date from JSON
    json_t *date_json = json_object_get(root, "date");
    if (!json_is_string(date_json)) {
        fprintf(stderr, "Error: date is not a string\n");
        json_decref(root);
        return 1;
    }

    const char *json_date = json_string_value(date_json);
    if (strcmp(current_date, json_date) != 0) {
        printf("Date does not match. Exiting...\n");
        json_decref(root);
        return 0;
    }

    // Extract symbols from JSON
    json_t *symbols = json_object_get(root, "symbol");
    if (!json_is_object(symbols)) {
        fprintf(stderr, "Error: symbol is not an object\n");
        json_decref(root);
        return 1;
    }

    // Iterate over each symbol
    const char *key;
    json_t *value;
    json_object_foreach(symbols, key, value) {
        json_t *action = json_object_get(value, "action");
        json_t *lot_size = json_object_get(value, "lot_size");

        if (!json_is_string(action) || !json_is_number(lot_size)) {
            fprintf(stderr, "Error: Invalid data for symbol %s\n", key);
            continue;
        }

        printf("Symbol: %s, Action: %s, Lot Size: %.2f\n", key, json_string_value(action), json_number_value(lot_size));
    }

    // Free the root object
    json_decref(root);

    return 0;
}
