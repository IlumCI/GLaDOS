/*
 * vkcompute -- one compute dispatch, checked on the CPU.
 *
 * vulkaninfo proves a driver enumerates; it submits almost nothing. This
 * records a command buffer, dispatches a shader over a million elements,
 * waits on a fence, and compares every element against what the CPU
 * computes. So a pass means commands went down the ring, the host's GPU
 * ran them, the fence came back through a sync file, and the host-visible
 * memory the result sits in is the memory this program reads.
 *
 *   vkcompute [elements]
 *
 * Exit 0 on a match, 1 on any mismatch, 2 when Vulkan refused something.
 *
 * Build (headers from KhronosGroup/Vulkan-Headers, not in this tree):
 *   glslc -fshader-stage=compute -mfmt=c -o out/vkcompute.spv.inc tools/guest/vkcompute.comp
 *   gcc -O2 -I<Vulkan-Headers>/include -Iout -o out/vkcompute tools/guest/vkcompute.c -lvulkan
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vulkan/vulkan.h>

static const uint32_t spirv[] =
#include "vkcompute.spv.inc"
    ;

#define CHECK(x)                                                             \
    do {                                                                     \
        VkResult r_ = (x);                                                   \
        if (r_ != VK_SUCCESS) {                                              \
            fprintf(stderr, "vkcompute: %s failed with %d\n", #x, (int)r_);  \
            return 2;                                                        \
        }                                                                    \
    } while (0)

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

int main(int argc, char **argv) {
    uint32_t n = argc > 1 ? (uint32_t)strtoul(argv[1], 0, 0) : 1u << 20;
    n = (n + 63) / 64 * 64;
    VkDeviceSize bytes = (VkDeviceSize)n * 4;

    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, 0, "vkcompute", 1, 0, 0, VK_API_VERSION_1_1};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, 0, 0, &app};
    VkInstance inst;
    CHECK(vkCreateInstance(&ici, 0, &inst));

    uint32_t ng = 1;
    VkPhysicalDevice pd;
    VkResult er = vkEnumeratePhysicalDevices(inst, &ng, &pd);
    if ((er != VK_SUCCESS && er != VK_INCOMPLETE) || ng == 0) {
        fprintf(stderr, "vkcompute: no physical device\n");
        return 2;
    }
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(pd, &props);
    printf("device  %s\n", props.deviceName);

    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, 0);
    VkQueueFamilyProperties qf[16];
    if (nq > 16) nq = 16;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qf);
    uint32_t fam = UINT32_MAX;
    for (uint32_t i = 0; i < nq; i++)
        if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { fam = i; break; }
    if (fam == UINT32_MAX) {
        fprintf(stderr, "vkcompute: no compute queue\n");
        return 2;
    }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, 0, 0, fam, 1, &prio};
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, 0, 0, 1, &qci};
    VkDevice dev;
    CHECK(vkCreateDevice(pd, &dci, 0, &dev));
    VkQueue q;
    vkGetDeviceQueue(dev, fam, 0, &q);

    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, 0, 0, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_SHARING_MODE_EXCLUSIVE};
    VkBuffer buf;
    CHECK(vkCreateBuffer(dev, &bci, 0, &buf));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev, buf, &req);
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    /* VKC_CACHED=1 asks for host-cached memory, which on NVIDIA is a
     * different heap path from the write-combined default. */
    uint32_t want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if (getenv("VKC_CACHED")) want |= VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    uint32_t mt = UINT32_MAX;
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((req.memoryTypeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) { mt = i; break; }
    if (mt == UINT32_MAX) {
        fprintf(stderr, "vkcompute: no host-visible coherent memory\n");
        return 2;
    }
    printf("memory    type %u, flags %#x\n", mt, mp.memoryTypes[mt].propertyFlags);
    VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, 0, req.size, mt};
    VkDeviceMemory mem;
    CHECK(vkAllocateMemory(dev, &mai, 0, &mem));
    CHECK(vkBindBufferMemory(dev, buf, mem, 0));
    uint32_t *p;
    CHECK(vkMapMemory(dev, mem, 0, bytes, 0, (void **)&p));
    memset(p, 0xa5, bytes);

    VkShaderModuleCreateInfo smci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, 0, 0, sizeof spirv, spirv};
    VkShaderModule sm;
    CHECK(vkCreateShaderModule(dev, &smci, 0, &sm));
    VkDescriptorSetLayoutBinding b0 = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT};
    VkDescriptorSetLayoutCreateInfo dslci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, 0, 0, 1, &b0};
    VkDescriptorSetLayout dsl;
    CHECK(vkCreateDescriptorSetLayout(dev, &dslci, 0, &dsl));
    VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 4};
    VkPipelineLayoutCreateInfo plci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, 0, 0, 1, &dsl, 1, &pcr};
    VkPipelineLayout pl;
    CHECK(vkCreatePipelineLayout(dev, &plci, 0, &pl));
    VkComputePipelineCreateInfo cpci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, 0, 0,
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, 0, 0, VK_SHADER_STAGE_COMPUTE_BIT, sm, "main"}, pl};
    VkPipeline pipe;
    CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, 0, &pipe));

    VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
    VkDescriptorPoolCreateInfo dpci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, 0, 0, 1, 1, &ps};
    VkDescriptorPool dp;
    CHECK(vkCreateDescriptorPool(dev, &dpci, 0, &dp));
    VkDescriptorSetAllocateInfo dsai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, 0, dp, 1, &dsl};
    VkDescriptorSet ds;
    CHECK(vkAllocateDescriptorSets(dev, &dsai, &ds));
    VkDescriptorBufferInfo dbi = {buf, 0, bytes};
    VkWriteDescriptorSet w = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, 0, ds, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 0, &dbi};
    vkUpdateDescriptorSets(dev, 1, &w, 0, 0);

    VkCommandPoolCreateInfo cpi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, 0, 0, fam};
    VkCommandPool cp;
    CHECK(vkCreateCommandPool(dev, &cpi, 0, &cp));
    VkCommandBufferAllocateInfo cbai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, 0, cp, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    VkCommandBuffer cb;
    CHECK(vkAllocateCommandBuffers(dev, &cbai, &cb));
    VkCommandBufferBeginInfo cbbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, 0, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    CHECK(vkBeginCommandBuffer(cb, &cbbi));
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, 0);
    vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &n);
    vkCmdDispatch(cb, n / 64, 1, 1);
    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, 0, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, 0, 0, 0);
    CHECK(vkEndCommandBuffer(cb));

    VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    CHECK(vkCreateFence(dev, &fci, 0, &fence));
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, 0, 0, 0, 0, 1, &cb};
    double t0 = now_ms();
    CHECK(vkQueueSubmit(q, 1, &si, fence));
    CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, 10ull * 1000 * 1000 * 1000));
    double t1 = now_ms();

    /* The shader writes i*i + 7, wrapping, which no fill pattern matches. */
    uint32_t bad = 0, first = UINT32_MAX;
    for (uint32_t i = 0; i < n; i++)
        if (p[i] != i * i + 7u) {
            if (first == UINT32_MAX) first = i;
            bad++;
        }
    printf("dispatch  %u elements, submit to fence %.2f ms\n", n, t1 - t0);
    if (bad) {
        /* Whether the result arrives late or never: a fence that signalled
         * early reads correct once the GPU catches up, a mapping that shows
         * the wrong memory stays wrong. */
        struct timespec ts = {0, 200 * 1000 * 1000};
        nanosleep(&ts, 0);
        uint32_t later = 0;
        for (uint32_t i = 0; i < n; i++)
            if (p[i] != i * i + 7u) later++;
        printf("after 200 ms, %u still wrong\n", later);
        printf("MISMATCH  %u of %u wrong, first at %u: %#x, wanted %#x\n", bad, n, first, p[first], first * first + 7u);
        return 1;
    }
    printf("match     every element is i*i+7, computed on the GPU\n");

    vkDestroyFence(dev, fence, 0);
    vkDestroyCommandPool(dev, cp, 0);
    vkDestroyDescriptorPool(dev, dp, 0);
    vkDestroyPipeline(dev, pipe, 0);
    vkDestroyPipelineLayout(dev, pl, 0);
    vkDestroyDescriptorSetLayout(dev, dsl, 0);
    vkDestroyShaderModule(dev, sm, 0);
    vkUnmapMemory(dev, mem);
    vkFreeMemory(dev, mem, 0);
    vkDestroyBuffer(dev, buf, 0);
    vkDestroyDevice(dev, 0);
    vkDestroyInstance(inst, 0);
    return 0;
}
