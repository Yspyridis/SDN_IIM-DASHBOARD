from django.db import models

# The below jsonfield raises error in saving the grid json
# from django.db.models import JSONField
from django.contrib.postgres.fields import ArrayField

# The below jsonfield saves the grid json correctly but will be depricated soon
from django.contrib.postgres.fields import JSONField
from django.utils.timezone import now


class IslandingScheme(models.Model):

    method_name                = models.CharField('Method', max_length=20)
    
    island_imbalance           = ArrayField(models.FloatField(blank=True, null=True), size=3, null=True)
    total_imbalance            = models.FloatField(blank=True, null=True)
    # island_imbalance           = ArrayField(models.FloatField(blank=True, null=True), size=3, null=True)
    # total_imbalance            = models.FloatField(blank=True, null=True)
    
    # island_imbalance_after_cut = ArrayField(models.FloatField(blank=True, null=True), size=3, null=True)
    island_imbalance_after_cut = ArrayField(models.FloatField(blank=True, null=True), null=True)
    total_imbalance_after_cut  = models.FloatField(blank=True, null=True)
    
    lines_to_cut               = models.IntegerField(blank=True, null=True)
        
    date                       = models.DateTimeField(default=now, editable=True)
    
    grid                       = JSONField(null=True)
    # grid                       = models.JSONField(default={}, null=True)