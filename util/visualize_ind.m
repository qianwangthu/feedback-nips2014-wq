function [] = visualize_ind(img, vis, ind, ite)
for guess = 1:10
    ite_rec{guess}(1, :) = img(ind, :);
    for n = 1:ite
        ite_rec{guess}(n+1, :) = squeeze(vis(guess, n, ind, :))';
    end
end
visualize_cell(ite_rec);
end