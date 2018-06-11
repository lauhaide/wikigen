--
-- Created by IntelliJ IDEA.
-- User: lperez
-- Date: 31/07/17
-- Time: 15:59
-- To change this template use File | Settings | File Templates.
--


function getWeightsName(weightsFile)
  local dn = string.split(weightsFile, "/")
  local i, _ = string.find(dn[#dn],".t7")
  return string.sub(dn[#dn],1,i-1)
end

function isDir(name)
    if type(name)~="string" then return false end
    local cd = lfs.currentdir()
    local is = lfs.chdir(name) and true or false
    lfs.chdir(cd)
    return is
end


