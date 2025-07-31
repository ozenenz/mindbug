# Simple training script for quick testing
from mindbug.training import train

if __name__ == "__main__":
    print("Running quick test training...")
    
    # Quick test configuration
    model, run_dir = train(config_name="quick", num_iterations=100)
    
    # For full training, uncomment the following line:
    # model, run_dir = train(config_name="performance", num_iterations=10000)
    
    print(f"Training complete! Model saved to: {run_dir}")