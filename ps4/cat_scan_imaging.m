clear

% Estimate matrix A
for n = 1:2500
    x = zeros(50, 50);
    x(floor((n-1)/50) + 1, mod(n-1, 50) + 1) = 1;
    
    A(:, n) = scanImage(x);
end

figure(1)
imshow(A, []);

% Get singular values of A with SVD
[U,S,V] = svd(A);

% Get the non-zero singular values and plot them
for i = 1:1950
    if S(i, i) <= 0
        break
    end
    sigma(i) = S(i, i);
end

i = 1:length(sigma);
figure(2)
plot(i, sigma);
title('Singular Values of A')
xlabel('Index i')
ylabel('Sigma(i)')

% From the plot, we estimate r = 1275
% Solve for x
y = scanImage();
r = 1275
x = V(:,1:r)*diag(1./diag(S(1:r,1:r)))*U(:,1:r)'*y;
x = reshape(x, [50, 50]);

figure(3)
imshow(x, []);