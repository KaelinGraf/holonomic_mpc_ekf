function v = getfield_default(s, name, default)
%GETFIELD_DEFAULT Return s.(name) if present and non-empty, otherwise default.
if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
    v = s.(name);
else
    v = default;
end
end
