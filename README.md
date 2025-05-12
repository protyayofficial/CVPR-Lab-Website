# Underwater Image Enhancement and Fish/Coral Detection Web Application 

## Project Description

This project provides a comprehensive solution for enhancing the quality of underwater images and performing fish/coral detection. It leverages various deep learning models and image processing techniques to address the challenges posed by underwater environments, such as color distortion, low contrast, and blurriness. The project includes different models for image enhancement and object detection, offering flexibility and options for various underwater imaging tasks.

## Features

*   **Underwater Image Enhancement:** Utilizes state-of-the-art models to improve the visual quality of underwater images.
*   **Fish Detection:** Implements object detection models specifically trained for identifying fish in underwater imagery.
*   **Coral Detection:** Implements object detection models specifically trained for identifying corals in underwater imagery.
*   **Multiple Models:** Includes several different models for both enhancement and detection, allowing for experimentation and comparison.
*   **Modular Design:** Organized into components and utility functions for better code structure and reusability.
*   **Easy Setup:** Provides a setup script for quickly getting the project running.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd <project_directory>
    ```

2.  **Set up the environment:**

    The project includes a `setup.sh` script and a `requirements.txt` file to facilitate the setup process. Execute the setup script:

    ```bash
    bash setup.sh
    ```

    This script will likely create a virtual environment and install the necessary dependencies listed in `requirements.txt`.

    Alternatively, you can manually create a virtual environment and install dependencies:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

## Usage

To run the main application, execute the `app.py` file:

```bash
streamlit run app.py
```

## License
This project is licensed under the terms of the [![MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE).

## Contributing
Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

* Fork the repository.
* Create a new branch for your feature or bugfix.
* Make your changes and commit them with clear messages.
* Push your changes to your fork.
* Submit a pull request to the main repository.

Please ensure your code adheres to the project's coding standards and includes appropriate tests.

## Contact
If you have any questions, suggestions, or issues, please feel free to open an issue on the GitHub repository or contact the project maintainers at [protyayofficial@gmail.com](mailto:protyayofficial@gmail.com).