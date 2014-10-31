function data_out = background_noise(data_in, noise_mask, scale, mask_type)
if nargin == 1
    scale = 1;
end
if ~exist('mask_type', 'var')
    mask_type = 'white';
end
noise_m = repmat(noise_mask, size(data_in, 1), 1);
if strcmp(mask_type, 'white')
    noise    = scale * rand(size(data_in)) .* noise_m;
    data_out = max(data_in, noise);
elseif strcmp(mask_type, 'black')
    noise    = ~noise_m;
    data_out = noise .* data_in;
end