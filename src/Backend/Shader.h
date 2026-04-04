#pragma once

#include <glm/glm.hpp>
#include <string>

class Shader {
public:
    Shader(const std::string& vertexPath, const std::string& fragmentPath);
    ~Shader();

    void use() const;
    unsigned int id() const;

    void setMat4(const std::string& name, const glm::mat4& value) const;
    void setVec4(const std::string& name, const glm::vec3& value) const;
    void setVec3(const std::string& name, const glm::vec3& value) const;
    void setFloat(const std::string& name, float value) const;

private:
    unsigned int programId;

    std::string loadFile(const std::string& path) const;
    unsigned int compile(unsigned int type, const std::string& source) const;
};