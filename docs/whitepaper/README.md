# Whitepaper Source

This directory contains the LaTeX source for the project whitepaper.

## Local Prerequisites

To compile the whitepaper locally, you must have the following installed and available in your system's PATH:

1.  **A TeX/LaTeX Distribution:** Such as [MiKTeX](https://miktex.org/) (Windows), [MacTeX](https://www.tug.org/mactex/) (macOS), or TeX Live (Linux, often via a package manager). The distribution must include `latexmk`.
2.  **`make`:** On Windows, `make` is available through tools like Chocolatey, MinGW/MSYS2, or the Windows Subsystem for Linux (WSL). On macOS and Linux, it is typically pre-installed or available via a package manager.
3.  **`git`:** Required for the version stamping in the footer.

## Building

With the prerequisites installed, you can build the PDF by running the following command from the root of the repository:

```sh
make all
```

This will generate the PDF and its SHA256 checksum in this directory. To clean up all generated files, run:

```sh
make clean
```

## Governance and Artifact Integrity

The whitepaper build process includes SHA256 checksum verification for all generated PDF artifacts. This is a critical step for governance, ensuring the integrity and immutability of released documentation. By providing a cryptographic hash, we can confirm that a whitepaper artifact has not been tampered with since its generation and official build.

This approach mirrors the Evidence Pack integrity principles used across other core systems. Just as Evidence Packs are signed and hashed to guarantee their authenticity and provenance, the whitepaper artifacts are similarly secured. This cryptographic attestation provides an audit-grade assurance for all versions of our foundational documentation.

This direct comparison to Evidence Pack Merkle roots is intentional; the checksum serves a similar purpose in creating an immutable, verifiable fingerprint of the whitepaper's content. Whitepapers, by their nature, lay out the foundational principles, design, and mechanisms of our systems, making them critical governance artifacts that require the highest level of integrity and verifiable authenticity. They are, in essence, the codified agreements of our architectural and operational philosophies.

A crucial implication of this cryptographic verification is that any alteration, no matter how minor, to the whitepaper's source (`.tex` files), the `Makefile`, or the build process itself, will result in a different SHA256 checksum for the generated PDF. Such a change effectively invalidates any prior citations based on an older checksum and requires the generation of a new, distinct checksum for the updated artifact. This ensures that every specific version of the whitepaper, as a governance artifact, is uniquely identifiable and auditable.
