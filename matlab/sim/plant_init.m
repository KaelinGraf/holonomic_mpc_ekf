function st = plant_init(pose0, vb0)
%PLANT_INIT Initial truth-plant state for plant_step.
%   st = PLANT_INIT(pose0, vb0)   pose0 [3x1] in the target frame, vb0 [3x1] body velocity (default zero)
if nargin < 2 || isempty(vb0), vb0 = zeros(3, 1); end
st.pose = pose0(:);
st.vb   = vb0(:);
st.w    = zeros(4, 1);
st.slip = zeros(2, 1);
end
