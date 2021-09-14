%% sin(2x)
x = 0:0.001:1;
y = sin(2*x);
fitlm(x,y)

% ===> OPTIMAL = 0.21975 +  0.97613 x


%% mean(sin(2x))
x1 = 0:0.001:1;
x2 = 0:0.001:1;

c = 0;
N = length(x1)*length(x2);
y = zeros(N,1);
X = zeros(N,2);
for i1=1:length(x1)
    for i2=1:length(x2)
        c = c + 1;
        X(c,1) = x1(i1);
        X(c,2) = x2(i2);
        
        y(c) = (sin(2*x1(i1)) + sin(2*x2(i2)))/2;
    end
end

fitlm(X,y)

% ===> OPTIMAL = 0.21975 +  0.48807 x1 + 0.48807 x2

%%
y = @(x) exp(x/10*(2*pi)).*sin(x*(2*pi));
f =@(x,theta) y(x)-sqrt(theta.^2-theta+1).*(sin(theta*x*(2*pi))+cos(theta*x*(2*pi)));
theta = linspace(-1,1,100000); 

% ===> OPTIMAL THETA = -0.178832  
 

for Nx=[1e2,1e3, 1e4]
    x = linspace(0,1,Nx)';

    ydata = y(x);
    mse = [];
    for j=1:length(theta)
        mse(j) = mean((f(x,theta(j)) - ydata).^2);
    end

    [opt, iopt] = min(mse);
    theta_opt = theta(iopt);

    plot(theta,mse,'-', 'Linewidth', 4)
    xline(theta_opt,'--', 'Linewidth', 4)
    fprintf('Nx %f: theta*=%f \n',Nx,theta_opt);
end
figure; plot(x,ydata, 'Linewidth',4)

%%
% This one is simpler:
y = @(x) 4*x+x.*sin(5*x); % target function
f =@(x,theta) x*theta; %mechanistic model.
theta = linspace(-5,10,100000);

% ===> OPTIMAL THETA = 3.565236 
figure;
for Nx=[1e2,1e3, 1e4]
    x = linspace(0,1,Nx)';

    ydata = y(x);
    mse = [];
    for j=1:length(theta)
        mse(j) = mean((f(x,theta(j)) - ydata).^2);
    end

    [opt, iopt] = min(mse);
    theta_opt = theta(iopt);

    plot(theta,mse,'-', 'Linewidth', 4)
    xline(theta_opt,'--', 'Linewidth', 4)
    fprintf('Nx %f: theta*=%f \n',Nx,theta_opt);
end
figure; plot(x,ydata, 'Linewidth',4)
