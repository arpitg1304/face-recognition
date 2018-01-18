function [lighting, required_sub, persons] = return_subset(im,subset,n, number, person)
sub_logical = subset ==n;
sub_indices = find(sub_logical);
s2 = size(sub_indices, 2);
required_sub = {};
lighting = [];
persons = [];
    for i = 1:s2
    required_sub{end+1} = im{sub_indices(i)};
    lighting = [lighting, number(sub_indices(i))];
    persons = [persons, person(sub_indices(i))];    
    end
end