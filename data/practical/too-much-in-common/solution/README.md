*   The challenge consists of two parts. The first part is vulnerable to a Common Modulus RSA attack.
*   Continuously request new ciphertexts from the server until two are found that share the same modulus `N`.
*   With two different public exponents and the corresponding ciphertexts for the same message under the same modulus, use the Extended Euclidean Algorithm to recover the original plaintext password.
*   For the second part, the server provides `N`, `e`, and `d`. Use the relationship between these values to find a multiple of `phi`.
*   This information allows for the factorization of `N` into its prime factors, `p` and `q`.
*   Finally, calculate `phi` directly from `p` and `q` and submit it to solve the challenge.