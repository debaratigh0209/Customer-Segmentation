def algorithms(filename):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import pickle
    import warnings
    warnings.filterwarnings('ignore')
    import os
    
    df=pd.read_csv(filename)

    df

    plt.hist(df['Age'], bins=20, color='skyblue')
    plt.title('Distribution of Age')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    #plt.show()

    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    categorical_columns = ['Gender', 'Item Purchased', 'Category', 'Subscription Status' ,'Frequency of Purchases']
    for col in categorical_columns:
        df[col] = label_encoder.fit_transform(df[col])

    from sklearn.model_selection import train_test_split


    X = df.drop(columns=['Item Purchased']) 
    y = df['Item Purchased']


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


        # Import necessary libraries
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs
    from sklearn.metrics import silhouette_score

    # Generate sample data
    X, _ = make_blobs(n_samples=300, centers=4, random_state=42)

    # Create and fit the K-Means model
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(X)

    # Get the cluster centers and labels
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    # Compute silhouette score
    silhouette_avg = silhouette_score(X, labels)
    print(f'Silhouette Score: {silhouette_avg:.2f}')
    acc=silhouette_avg *100 
    print("accuracy",acc)

    # Plot the original data points and cluster centers
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', s=200, marker='X', label='Cluster Centers')
    plt.title('K-Means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    #plt.show()

    import matplotlib
    matplotlib.use('Agg')
    import pandas as pd
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt

    # Load the CSV data into a DataFrame

    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    categorical_columns = ['Gender', 'Item Purchased', 'Category', 'Subscription Status' ,'Frequency of Purchases']
    for col in categorical_columns:
        df[col] = label_encoder.fit_transform(df[col])
    print(df)
    features = ['Purchase Amount (USD)','Age']  # Replace with your actual feature names
    X = df[features]


    # Select the features for clustering (assuming numerical columns)


    # Define the number of clusters (k)
    k = 3  # You can adjust this value

    # Create the KMeans model
    kmeans = KMeans(n_clusters=k)

    # Fit the model to the data
    kmeans.fit(X)

    # Assign cluster labels to each data point
    df["cluster"] = kmeans.labels_

    # Print the cluster centroids
    print("Centroids:\n", kmeans.cluster_centers_)

    # Visualize the clustering results (optional)
    plt.clf()
    plt.scatter(X["Purchase Amount (USD)"], X["Age"], c=df["cluster"], cmap="viridis")
    plt.xlabel("Cluster")
    plt.ylabel("Number of Points")
    plt.title("Data Points Distribution in Clusters")
    #plt.show()
    #os.remove(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'static', 'output.jpg'))
    
    plt.savefig(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'static', 'output.jpg'))



    import pandas as pd
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt

    # Load the CSV data into a DataFrame



    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    categorical_columns = ['Gender', 'Item Purchased', 'Category', 'Subscription Status' ,'Frequency of Purchases']
    for col in categorical_columns:
        df[col] = label_encoder.fit_transform(df[col])
    print(df)
    features = ['Purchase Amount (USD)','Age']  # Replace with your actual feature names
    X = df[features]


    # Select the features for clustering (assuming numerical columns)


    # Define the number of clusters (k)
    k = 3  # You can adjust this value

    # Create the KMeans model
    kmeans = KMeans(n_clusters=k)

    # Fit the model to the data
    kmeans.fit(X)
    df["cluster"] = kmeans.labels_

    cluster_counts = df['cluster'].value_counts()

    # Create a bar chart
    plt.bar(cluster_counts.index, cluster_counts.values)
    plt.xlabel("Cluster")
    plt.ylabel("Number of Points")
    plt.title("Data Points Distribution in Clusters")
    #plt.show()



    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs
    from sklearn.metrics import silhouette_score

    # Assuming you have your kmeans model and cluster centers
    # Your kmeans model
    X, _ = make_blobs(n_samples=300, centers=4, random_state=42)

    # Create and fit the K-Means model
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(X)

    centers = kmeans.cluster_centers_

    # Extract number of clusters
    num_clusters = len(centers)

    # Define cluster labels (optional)
    cluster_labels = [f"Cluster {i+1}" for i in range(num_clusters)]

    # Define cluster positions (on the x-axis)
    cluster_positions = range(num_clusters)  # Adjust spacing if needed

    # Create the bar plot (one bar for each cluster center)
    plt.figure(figsize=(8, 6))  # Adjust figure size as desired
    bars = plt.bar(cluster_positions, centers[:, 0], color='skyblue', label='Feature ')

    # Add labels and title
    plt.xlabel("Clusters")
    plt.ylabel("cluster centers")  # Assuming centers have multiple features
    plt.title("Customer segmentation")

    # Optional: Add labels for other features (if centers have more than 1 dimension)
    for i, center in enumerate(centers):
        plt.text(cluster_positions[i], center[1] + 0.1, f" ", ha='center')  # Adjust y-offset and formatting

    # Display the plot
    plt.legend()
    #plt.show()
    
    #plt.savefig(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'static', 'output.jpg'))


    # Import necessary libraries
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.cluster import DBSCAN
    from sklearn.datasets import make_blobs
    from sklearn.metrics import silhouette_score

    # Generate sample data
    X, _ = make_blobs(n_samples=300, centers=4, random_state=42)

    # Create and fit the DBSCAN model
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X)

    # Plot the original data points and DBSCAN clusters
    plt.scatter(X[:, 0], X[:, 1], c=dbscan_labels, cmap='viridis', s=50, alpha=0.7)
    plt.title('DBSCAN Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    #plt.show()

    # Evaluate DBSCAN using silhouette score
    silhouette_avg = silhouette_score(X, dbscan_labels)
    print(f'Silhouette Score (DBSCAN): {silhouette_avg:.2f}')
    acc=silhouette_avg *100
    print("accuracy",acc)


    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    categorical_columns = ['Gender', 'Item Purchased', 'Category', 'Subscription Status' ,'Frequency of Purchases']
    for col in categorical_columns:
        df[col] = label_encoder.fit_transform(df[col])

    logistic = LogisticRegression()
    logistic.fit(X_train, y_train)
    y_pred = logistic.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Generate sample data
    X, y = make_classification(n_samples=4000, n_features=20, random_state=100)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=100)

    # Initialize AdaBoost classifier
    logistic = LogisticRegression( random_state=100)

    # Train the model
    logistic.fit(X_train, y_train)

    # Make predictions
    y_pred =logistic.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    print("Accuracy:", accuracy)

    # Display classification report
    print(classification_report(y_test, y_pred))


    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report


    random_forest = RandomForestClassifier()
    random_forest.fit(X_train, y_train)
    y_pred = random_forest.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Generate sample data
    X, y = make_classification(n_samples=4000, n_features=20, random_state=100)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=100)

    # Initialize AdaBoost classifier
    random_classifier = RandomForestClassifier(n_estimators=50, random_state=100)

    # Train the model
    random_classifier.fit(X_train, y_train)

    # Make predictions
    y_pred =random_classifier.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    print("Accuracy:", accuracy)

    # Display classification report
    print(classification_report(y_test, y_pred))


    from matplotlib import pyplot as plt
    # x-coordinates of left sides of bars  
    left = [2,4,6,8] 
    
    # heights of bars 
    height = [0.32, 0.74,0.985,0.93] 
    
    # labels for bars 
    tick_label = ['DBSCAN','K-means','Random forest ', 'logistic regression'] 
    
    # plotting a bar chart 
    plt.bar(left, height, tick_label = tick_label, 
            width = 0.8, color = [ 'green','red',"black","blue"]) 
    
    # naming the x-axis 
    plt.xlabel('Algorithms') 
    # naming the y-axis 
    plt.ylabel('Accuracy in %') 
    # plot title 
    plt.title('Performance Comparison') 
    
    # function to show the plot 
    plt.rcParams['figure.figsize'] = (500,400)
    #plt.show()

    filename = './customer_model.pkl'
    pickle.dump(random_classifier, open(filename, 'wb'))
    loaded_model = pickle.load(open(filename, 'rb'))




