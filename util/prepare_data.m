function [train_x, train_y, test_x, test_y] = prepare_data(filename, ind_train, ind_test)
load(filename);
train_x = double(train_x);
test_x  = double(test_x);
train_x = train_x / max(max(train_x));
test_x  = test_x  / max(max(test_x));

train_y = double(train_y);
test_y  = double(test_y);

x   = [train_x; test_x];
y   = [train_y; test_y];

train_x = x(ind_train, :);
test_x  = x(ind_test, :);
train_y = y(ind_train, :);
test_y  = y(ind_test, :);
end