% Parameters
beta = 1.5; % Infection rate
gamma = 1/7; % Recovery rate
N = 1000; % Total population
I0 = 1; % Initial number of infected individuals
S0 = N - I0; % Initial number of susceptible individuals
R0 = 0; % Initial number of recovered individuals

% Initial state
S = S0;
I = I0;
R = R0;
t = 0; % Initial time
T_end = 100; % Simulation end time
data = [t, S, I, R]; % Record the initial state

% Main simulation loop (Gillespie algorithm)
while t < T_end && I > 0
    % Calculate propensities
    a_infection = beta * S * I / N;
    a_recovery = gamma * I;
    a_total = a_infection + a_recovery;
    
    % Time to next event
    tau = -log(rand) / a_total;
    t = t + tau;
    
    % Determine which event occurs
    if rand < a_infection / a_total
        % Infection event
        S = S - 1;
        I = I + 1;
    else
        % Recovery event
        I = I - 1;
        R = R + 1;
    end
    
    % Record the state
    data = [data; t, S, I, R];
end

% Plot the results
plot(data(:,1), data(:,2), 'g', data(:,1), data(:,3), 'r', data(:,1), data(:,4), 'k');
legend('Susceptible', 'Infected', 'Recovered');
xlabel('Time');
ylabel('Population');
title('SIR Model Simulation using Gillespie Algorithm');
