// Quiz functionality for DeepFake Detection App

document.addEventListener('DOMContentLoaded', function() {
    // Quiz configuration
    const quizQuestions = [
        {
            question: "What is a deepfake?",
            options: [
                "A type of computer virus",
                "AI-generated or manipulated media that replaces someone's likeness",
                "A security vulnerability in facial recognition",
                "A technique for enhancing low-quality images"
            ],
            correctAnswer: 1,
            explanation: "Deepfakes use artificial intelligence to create or manipulate media, typically replacing someone's face or voice with someone else's in a convincing way."
        },
        {
            question: "Which technology is primarily used to create deepfakes?",
            options: [
                "Blockchain",
                "Virtual Reality",
                "Deep Learning / Neural Networks",
                "Quantum Computing"
            ],
            correctAnswer: 2,
            explanation: "Deepfakes are created using deep learning techniques, particularly Generative Adversarial Networks (GANs) or autoencoders, which are types of neural networks."
        },
        {
            question: "What does MesoNet focus on when detecting deepfakes?",
            options: [
                "Audio inconsistencies",
                "Metadata analysis",
                "Mesoscopic properties of images",
                "Blockchain verification"
            ],
            correctAnswer: 2,
            explanation: "MesoNet analyzes mesoscopic properties of images - the features that exist between the microscopic pixel level and the macroscopic image level."
        },
        {
            question: "Which of these is NOT a common sign of a deepfake?",
            options: [
                "Unnatural eye reflections",
                "Blurry or changing facial features",
                "Unusual skin texture",
                "Visible watermarks"
            ],
            correctAnswer: 3,
            explanation: "Watermarks are typically added intentionally to mark media ownership. They aren't inherent artifacts of deepfake generation."
        },
        {
            question: "Why are convolutional neural networks (CNNs) effective for deepfake detection?",
            options: [
                "They can process images very quickly",
                "They can identify patterns and features at different scales",
                "They require less computational power",
                "They work directly with the image file metadata"
            ],
            correctAnswer: 1,
            explanation: "CNNs excel at identifying visual patterns and features at different scales, making them ideal for detecting the subtle artifacts and inconsistencies in deepfakes."
        },
        {
            question: "Which part of a face is typically most difficult for deepfake algorithms to render realistically?",
            options: [
                "Nose",
                "Hair",
                "Eyes and teeth",
                "Cheeks"
            ],
            correctAnswer: 2,
            explanation: "Eyes and teeth are often the most challenging features for deepfake algorithms to render realistically. Look for unnatural reflections in eyes and odd-looking teeth."
        },
        {
            question: "How many convolutional layers does the basic MesoNet architecture use?",
            options: [
                "2",
                "4",
                "8",
                "16"
            ],
            correctAnswer: 1,
            explanation: "The basic MesoNet architecture uses 4 convolutional layers, making it relatively compact but still effective for deepfake detection."
        },
        {
            question: "What is a 'kernel' in the context of convolutional neural networks?",
            options: [
                "The central processing unit of the neural network",
                "A small matrix that slides over the input image to extract features",
                "A type of deepfake technique",
                "The final decision-making layer"
            ],
            correctAnswer: 1,
            explanation: "A kernel (or filter) is a small matrix that slides over the input image, performing convolution operations to extract features at each position."
        },
        {
            question: "What does 'pooling' do in a CNN like MesoNet?",
            options: [
                "Increases the image resolution",
                "Combines multiple images together",
                "Reduces the spatial dimensions and creates position invariance",
                "Adds more color information"
            ],
            correctAnswer: 2,
            explanation: "Pooling reduces the spatial dimensions (width and height) of the data, helping to create position invariance and reducing computational load."
        },
        {
            question: "Which of these is NOT a potential danger of deepfake technology?",
            options: [
                "Political misinformation",
                "Identity theft",
                "Non-consensual synthetic media",
                "Improved cybersecurity"
            ],
            correctAnswer: 3,
            explanation: "While deepfakes pose many risks, they don't inherently improve cybersecurity. In fact, they often represent a cybersecurity threat."
        }
    ];
    
    // Quiz elements
    const startButton = document.getElementById('start-quiz');
    const restartButton = document.getElementById('restart-quiz');
    const quizStart = document.getElementById('quiz-start');
    const questionsContainer = document.getElementById('questions-container');
    const quizResults = document.getElementById('quiz-results');
    const progressBar = document.querySelector('#quiz-progress .progress-bar');
    const resultProgressBar = document.getElementById('result-progress-bar');
    const scoreDisplay = document.getElementById('score-display');
    const scoreMessage = document.getElementById('score-message');
    
    // Quiz state
    let currentQuestion = 0;
    let score = 0;
    let quizStarted = false;
    
    // Start quiz event
    if (startButton) {
        startButton.addEventListener('click', startQuiz);
    }
    
    // Restart quiz event
    if (restartButton) {
        restartButton.addEventListener('click', restartQuiz);
    }
    
    function startQuiz() {
        quizStarted = true;
        currentQuestion = 0;
        score = 0;
        
        // Hide start screen, show questions
        quizStart.classList.add('d-none');
        questionsContainer.classList.remove('d-none');
        
        // Load first question
        loadQuestion();
    }
    
    function restartQuiz() {
        // Hide results, show start screen
        quizResults.classList.add('d-none');
        quizStart.classList.remove('d-none');
        
        // Reset progress bar
        updateProgress(0);
    }
    
    function loadQuestion() {
        // Clear previous question
        questionsContainer.innerHTML = '';
        
        if (currentQuestion < quizQuestions.length) {
            const question = quizQuestions[currentQuestion];
            
            // Create question card
            const questionCard = document.createElement('div');
            questionCard.className = 'question-card active';
            
            // Question header
            const questionHeader = document.createElement('h4');
            questionHeader.className = 'mb-4';
            questionHeader.textContent = `Question ${currentQuestion + 1}: ${question.question}`;
            
            // Options container
            const optionsContainer = document.createElement('div');
            optionsContainer.className = 'options-container';
            
            // Create option buttons
            question.options.forEach((option, index) => {
                const optionButton = document.createElement('button');
                optionButton.className = 'option-btn';
                optionButton.setAttribute('data-index', index);
                optionButton.textContent = option;
                
                optionButton.addEventListener('click', () => selectOption(index));
                
                optionsContainer.appendChild(optionButton);
            });
            
            // Feedback container
            const feedbackContainer = document.createElement('div');
            feedbackContainer.className = 'feedback mt-3';
            feedbackContainer.id = 'feedback';
            
            // Next button
            const nextButton = document.createElement('button');
            nextButton.className = 'btn btn-primary mt-3 d-none';
            nextButton.id = 'next-button';
            nextButton.textContent = currentQuestion === quizQuestions.length - 1 ? 'Finish Quiz' : 'Next Question';
            nextButton.addEventListener('click', nextQuestion);
            
            // Assemble question card
            questionCard.appendChild(questionHeader);
            questionCard.appendChild(optionsContainer);
            questionCard.appendChild(feedbackContainer);
            questionCard.appendChild(nextButton);
            
            // Add to container
            questionsContainer.appendChild(questionCard);
            
            // Update progress
            updateProgress((currentQuestion / quizQuestions.length) * 100);
        } else {
            // No more questions, show results
            showResults();
        }
    }
    
    function selectOption(index) {
        // Prevent selection after answer is revealed
        if (document.getElementById('next-button').classList.contains('d-none') === false) {
            return;
        }
        
        const question = quizQuestions[currentQuestion];
        const optionButtons = document.querySelectorAll('.option-btn');
        const feedbackContainer = document.getElementById('feedback');
        const nextButton = document.getElementById('next-button');
        
        // Clear previous selections
        optionButtons.forEach(button => {
            button.classList.remove('selected', 'correct', 'incorrect');
        });
        
        // Mark selected option
        optionButtons[index].classList.add('selected');
        
        // Check if correct
        const isCorrect = index === question.correctAnswer;
        
        // Update score
        if (isCorrect) {
            score++;
        }
        
        // Show correct/incorrect styling
        optionButtons[index].classList.add(isCorrect ? 'correct' : 'incorrect');
        optionButtons[question.correctAnswer].classList.add('correct');
        
        // Show feedback
        feedbackContainer.textContent = question.explanation;
        feedbackContainer.classList.add(isCorrect ? 'alert-success' : 'alert-danger');
        feedbackContainer.style.display = 'block';
        
        // Show next button
        nextButton.classList.remove('d-none');
    }
    
    function nextQuestion() {
        currentQuestion++;
        loadQuestion();
    }
    
    function showResults() {
        // Hide questions, show results
        questionsContainer.classList.add('d-none');
        quizResults.classList.remove('d-none');
        
        // Calculate and display score
        const percentage = Math.round((score / quizQuestions.length) * 100);
        scoreDisplay.textContent = `${percentage}%`;
        resultProgressBar.style.width = `${percentage}%`;
        resultProgressBar.setAttribute('aria-valuenow', percentage);
        
        // Color the progress bar
        if (percentage >= 80) {
            resultProgressBar.className = 'progress-bar bg-success';
            scoreMessage.textContent = 'Excellent! You have a strong understanding of deepfakes.';
        } else if (percentage >= 60) {
            resultProgressBar.className = 'progress-bar bg-info';
            scoreMessage.textContent = 'Good job! You have a solid grasp of the basics.';
        } else if (percentage >= 40) {
            resultProgressBar.className = 'progress-bar bg-warning';
            scoreMessage.textContent = 'Not bad, but there\'s room for improvement.';
        } else {
            resultProgressBar.className = 'progress-bar bg-danger';
            scoreMessage.textContent = 'You might want to review the educational materials and try again.';
        }
    }
    
    function updateProgress(percent) {
        progressBar.style.width = `${percent}%`;
        progressBar.setAttribute('aria-valuenow', percent);
        progressBar.textContent = `${Math.round(percent)}%`;
    }
});
