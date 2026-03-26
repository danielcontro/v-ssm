# Value Functions as Supermartingale Certificates

To reproduce the experiments make sure to follow the instructions below.

- Make sure that you have Python 3.13 or higher installed on your system.

- Make sure that the dependencies listed in `requirements.txt` are installed (the use of a virtual environment is strongly recommended).

- The certificate synthesis can be reproduced with the following command:

```bash
python -m scripts.certificate_synthesis
```

- Make sure that the [PRISM model checker](https://www.prismmodelchecker.org/) is installed and available on your system.

- The PRISM verification can be reproduced with the following command:

```bash
python -m scripts.prism_verification --prism_binary /path/to/prism --timeout timeout_in_seconds
```
