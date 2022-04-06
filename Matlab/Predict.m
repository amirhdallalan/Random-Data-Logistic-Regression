function p = Predict(theta, X)

    h_x = Sigmoid(X * theta);
    p = (h_x >= 0.5);

end