import skore

def view_results():
    print("=== Accessing Skore Project ===")
    project = skore.Project("reports/titanic_skore_project.skore", if_exists='load')
    
    available_keys = list(project.keys())
    
    if not available_keys:
        print("The database is empty!")
        return

    report_name = available_keys[0]
    print(f"Fetching metrics for: '{report_name}'...\n")
    
    try:
        report = project.get(report_name)
        
        print("--- Accuracy ---")
        print(report.metrics.accuracy())
        
        print("\n--- Precision ---")
        print(report.metrics.precision())
        
        print("\n--- Recall ---")
        print(report.metrics.recall())
        
        
    except Exception as e:
        print(f"Error loading the report: {e}")

if __name__ == "__main__":
    view_results()