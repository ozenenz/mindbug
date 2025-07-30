from mindbug.training import train

if __name__ == "__main__":
    print("Running quick test training...")
    model, run_dir = train(config_name="quick", num_iterations=100)
    # model, run_dir = train(config_name="performance", num_iterations=10000)
