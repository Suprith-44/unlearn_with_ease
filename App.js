import React, { useState } from 'react';
import './App.css';

function App() {
  const [currentPage, setCurrentPage] = useState('welcome');

  const WelcomePage = () => (
    <div style={pageStyle}>
      <h1 style={titleStyle}>Welcome to unlearn_with_ease</h1>
      
      <p style={descriptionStyle}>
        Revolutionizing machine learning with selective unlearning.
        Remove specific data from trained models without full retraining.
      </p>
      
      <div style={featureBoxStyle}>
        <p>
          unlearn_with_ease is a cutting-edge MLaaS platform that allows you to:
        </p>
        <ul style={{textAlign: 'left'}}>
          <li>Upload trained models</li>
          <li>Specify data for removal</li>
          <li>Unlearn selected data efficiently</li>
          <li>Preserve model integrity</li>
        </ul>
      </div>
      
      <button onClick={() => setCurrentPage('login')} style={buttonStyle}>
        Login
      </button>
    </div>
  );

  const LoginPage = () => (
    <div style={pageStyle}>
      <div style={formContainerStyle}>
        <h2 style={{textAlign: 'center', marginBottom: '1rem'}}>Login</h2>
        <input type="text" placeholder="Username" style={inputStyle} />
        <input type="password" placeholder="Password" style={inputStyle} />
        <button onClick={() => setCurrentPage('main')} style={buttonStyle}>Login</button>
        <p style={{textAlign: 'center', marginTop: '1rem'}}>
          Don't have an account? <button onClick={() => setCurrentPage('signup')} style={linkStyle}>Sign up</button>
        </p>
      </div>
    </div>
  );

  const SignupPage = () => (
    <div style={pageStyle}>
      <div style={formContainerStyle}>
        <h2 style={{textAlign: 'center', marginBottom: '1rem'}}>Sign Up</h2>
        <input type="text" placeholder="Name" style={inputStyle} />
        <input type="email" placeholder="Email" style={inputStyle} />
        <input type="text" placeholder="Username" style={inputStyle} />
        <input type="password" placeholder="Password" style={inputStyle} />
        <button style={buttonStyle}>Create Account</button>
        <p style={{textAlign: 'center', marginTop: '1rem'}}>
          Already have an account? <button onClick={() => setCurrentPage('login')} style={linkStyle}>Login</button>
        </p>
      </div>
    </div>
  );

  const MainPage = () => {
    const [modelFile, setModelFile] = useState(null);
    const [classesToUnlearn, setClassesToUnlearn] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
    const [result, setResult] = useState(null);
    const [progress, setProgress] = useState(0);
  
    const handleModelUpload = (event) => {
      setModelFile(event.target.files[0]);
    };
  
    const handleSubmit = async () => {
      if (!modelFile || !classesToUnlearn) {
        setError('Please upload a model file and specify classes to unlearn.');
        return;
      }
  
      setIsLoading(true);
      setError(null);
      setResult(null);
      setProgress(0);
  
      const formData = new FormData();
      formData.append('model', modelFile);
      formData.append('classes', classesToUnlearn);
  
      try {
        const response = await fetch('http://localhost:5000/unlearn', {
          method: 'POST',
          body: formData,
        });
  
        if (response.ok) {
          const { task_id } = await response.json();
          
          // Poll for progress
          const pollInterval = setInterval(async () => {
            const progressResponse = await fetch(`http://localhost:5000/progress/${task_id}`);
            const progressData = await progressResponse.json();
  
            if (progressResponse.status === 200 && 'progress' in progressData) {
              setProgress(progressData.progress);
            } else if (progressResponse.status === 200) {
              // Task completed
              clearInterval(pollInterval);
              setResult(progressData);
              setIsLoading(false);
  
              // Handle model download
              const blob = new Blob([progressData.unlearned_model], {type: 'application/octet-stream'});
              const url = URL.createObjectURL(blob);
              const a = document.createElement('a');
              a.href = url;
              a.download = 'unlearned_model.pt';
              document.body.appendChild(a);
              a.click();
              document.body.removeChild(a);
            } else if (progressResponse.status === 500) {
              clearInterval(pollInterval);
              setError(progressData.error);
              setIsLoading(false);
            }
          }, 1000); // Poll every second
        } else {
          setError('Failed to start unlearning task. Please try again.');
          setIsLoading(false);
        }
      } catch (error) {
        console.error('Error:', error);
        setError('An error occurred. Please try again.');
        setIsLoading(false);
      }
    };
  
    return (
      <div style={pageStyle}>
        <h1 style={titleStyle}>unlearn_with_ease</h1>
        
        <p style={descriptionStyle}>
          Welcome to unlearn_with_ease, your platform for selective machine learning model unlearning.
          Upload your model and specify the classes you want to unlearn.
        </p>
        
        <div style={uploadContainerStyle}>
          <div style={uploadBoxStyle}>
            <h3>Upload ML Model</h3>
            <input 
              type="file" 
              onChange={handleModelUpload} 
              style={fileInputStyle} 
              disabled={isLoading}
            />
          </div>
          <div style={uploadBoxStyle}>
            <h3>Classes to Unlearn</h3>
            <input 
              type="text" 
              placeholder="Enter classes separated by commas"
              value={classesToUnlearn}
              onChange={(e) => setClassesToUnlearn(e.target.value)}
              style={inputStyle}
              disabled={isLoading}
            />
          </div>
        </div>
        
        <button 
          onClick={handleSubmit} 
          style={buttonStyle} 
          disabled={isLoading}
        >
          {isLoading ? 'Processing...' : 'Submit for Unlearning'}
        </button>
  
        {isLoading && (
          <div style={messageStyle}>
            Unlearning in progress: {progress}%
          </div>
        )}
  
        {error && (
          <div style={errorStyle}>
            {error}
          </div>
        )}
  
        {result && (
          <div style={resultStyle}>
            <h3>Unlearning Results:</h3>
            <p>Forget Accuracy: {result.forget_accuracy.toFixed(2)}%</p>
            <p>Forget Loss: {result.forget_loss.toFixed(4)}</p>
            <p>Retain Accuracy: {result.retain_accuracy.toFixed(2)}%</p>
            <p>Retain Loss: {result.retain_loss.toFixed(4)}</p>
            <p>Unlearned model download should start automatically.</p>
          </div>
        )}
      </div>
    );
  };
  
  // Additional styles
  const messageStyle = {
    marginTop: '1rem',
    color: '#FFD700',
    fontWeight: 'bold',
  };
  
  const errorStyle = {
    marginTop: '1rem',
    color: '#FF6347',
    fontWeight: 'bold',
  };
  
  const resultStyle = {
    marginTop: '1rem',
    background: 'rgba(255, 255, 255, 0.1)',
    padding: '1rem',
    borderRadius: '10px',
  };

  // Styles
  const pageStyle = {
    minHeight: '100vh',
    background: 'linear-gradient(45deg, #1a1a2e, #16213e, #0f3460)',
    backgroundSize: '400% 400%',
    animation: 'gradientBG 15s ease infinite',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    color: 'white',
    textAlign: 'center',
    padding: '2rem'
  };

  const titleStyle = {
    fontSize: '3rem',
    marginBottom: '1rem',
    animation: 'fadeIn 2s ease-in'
  };

  const descriptionStyle = {
    fontSize: '1.2rem',
    maxWidth: '600px',
    marginBottom: '2rem',
    animation: 'fadeIn 2s ease-in 0.5s both'
  };

  const featureBoxStyle = {
    background: 'rgba(255, 255, 255, 0.1)',
    padding: '2rem',
    borderRadius: '10px',
    marginBottom: '2rem',
    animation: 'fadeIn 2s ease-in 1s both'
  };

  const formContainerStyle = {
    background: 'rgba(255, 255, 255, 0.1)',
    padding: '2rem',
    borderRadius: '10px',
    width: '300px',
    animation: 'fadeIn 0.5s ease-in'
  };

  const inputStyle = {
    width: '100%',
    padding: '10px',
    margin: '10px 0',
    borderRadius: '5px',
    border: 'none',
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    color: 'white'
  };

  const buttonStyle = {
    padding: '10px 20px',
    margin: '10px 0',
    borderRadius: '5px',
    border: 'none',
    backgroundColor: '#e94560',
    color: 'white',
    cursor: 'pointer',
    transition: 'all 0.3s ease',
    fontSize: '1.2rem'
  };

  const linkStyle = {
    background: 'none',
    border: 'none',
    color: '#e94560',
    textDecoration: 'underline',
    cursor: 'pointer'
  };

  const uploadContainerStyle = {
    display: 'flex',
    justifyContent: 'space-around',
    width: '100%',
    maxWidth: '800px',
    marginBottom: '2rem'
  };

  const uploadBoxStyle = {
    background: 'rgba(255, 255, 255, 0.1)',
    padding: '1rem',
    borderRadius: '10px',
    width: '45%'
  };

  const fileInputStyle = {
    margin: '1rem 0'
  };

  return (
    <div className="App">
      {currentPage === 'welcome' && <WelcomePage />}
      {currentPage === 'login' && <LoginPage />}
      {currentPage === 'signup' && <SignupPage />}
      {currentPage === 'main' && <MainPage />}
      
      <style>{`
        @keyframes gradientBG {
          0% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
          100% { background-position: 0% 50%; }
        }
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(20px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  );
}

export default App;