function r=visualize_cell(X, mm, s1, s2)
%%% X should be 1*N cell, every cell have same size matrix

num_group        = numel(X);
[num_sample, D]  = size(X{1});
s = sqrt(D);
b = 1;
a = ones(num_group * s + num_group *b , num_sample * s + num_sample);
x = 0; y = 0;
for n = 1:num_group
    for m = 1:num_sample
        im = reshape(X{n}(m, :), [s, s]);
        a(x*s+1 + x*b: x*s + s + x*b, y*s+1+y : y*s+s+y) = im';
        y = y+1;
    end
    x = x+1;
    y = 0;
end
imshow(a, [min(a(:)), max(a(:))]);


% 
% %FROM RBMLIB http://code.google.com/p/matrbm/
% %Visualize weights X. If the function is called as a void method,
% %it does the plotting. But if the function is assigned to a variable 
% %outside of this code, the formed image is returned instead.
% if ~exist('mm','var')
%     mm = [min(X(:)) max(X(:))];
% end
% if ~exist('s1','var')
%     s1 = 0;
% end
% if ~exist('s2','var')
%     s2 = 0;
% end
%     
% [D,N]= size(X);
% s=sqrt(D);
% if s==floor(s) || (s1 ~=0 && s2 ~=0)
%     if (s1 ==0 || s2 ==0)
%         s1 = s; s2 = s;
%     end
%     %its a square, so data is probably an image
%     num=ceil(sqrt(N));
%     a=mm(2)*ones(num*s2+num-1,num*s1+num-1);
%     x=0;
%     y=0;
%     for i=1:N
%         %%%% for uint8 %%%%%
% % %         im = reshape(X(:,i),s1,s2)';
%         %%%% for noise %%%%%
%         im = reshape(X(:,i),s1,s2);
%         a(x*s2+1+x : x*s2+s2+x, y*s1+1+y : y*s1+s1+y)=im';
%         x=x+1;
%         if(x>=num)
%             x=0;
%             y=y+1;
%         end
%     end
%     d=true;
% else
%     %there is not much we can do
%     a=X;
% end
% 
% %return the image, or plot the image
% if nargout==1
%     r=a;
% else
%     imshow(a, [mm(1) mm(2)]);
% end