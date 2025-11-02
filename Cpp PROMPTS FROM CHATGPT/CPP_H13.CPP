#include <iostream>
#include <string>
#include <cryptx>

using namespace std;
using namespace cryptx;

int main() {
    // Initialize AES-256 encryptor
    AES256 encryptor;

    // Set initial key
    encryptor.setKey("my_super_secret_32_byte_key_for_AES!");

    // Example plaintext
    string plaintext = "This is a top secret message.";

    // Encrypt the plaintext
    string ciphertext = encryptor.encrypt(plaintext);
    cout << "Encrypted: " << ciphertext << endl;

    // Rotate the key automatically
    encryptor.rotateKey();
    cout << "Key rotated." << endl;

    // Decrypt using the new key
    string decrypted = encryptor.decrypt(ciphertext);
    cout << "Decrypted: " << decrypted << endl;

    return 0;
}
